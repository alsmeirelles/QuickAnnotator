import base64
import glob
import os, re, sys, ast, shutil
from datetime import datetime
import logging
import multiprocessing

import PIL.Image
import cv2
import numpy as np
from skimage.measure import label
from flask import Blueprint, send_from_directory, jsonify
from flask import current_app, url_for, request, make_response

import sqlalchemy
import json

from QA_config import config, get_database_uri
from QA_db import Image, Project, Roi, db, Job, get_latest_modelid
from QA_pool import pool_get_image, pool_run_script, update_completed_job_status
from QA_utils import get_file_tail,tile_for_patch,get_initial_train,get_img_metadata,run_al

api = Blueprint("api", __name__)
jobs_logger = logging.getLogger('jobs')

# This will get the last few lines from the log file
@api.route("/api/logs/<file_stem>", methods=["GET"])
def get_latest_log(file_stem):
    log_lines = 100  # <-- TODO: pull from config
    log_path = file_stem + '.log'
    return get_file_tail(log_path, log_lines), 200


@api.route("/api/<project_name>/embed/<image_name>", methods=["GET"])
def get_embed(project_name, image_name):
    upload_folder = f"./projects/{project_name}/patches/"
    return send_from_directory(upload_folder, image_name)

@api.route("/api/<project_name>/generate_train", methods=["GET"])
def generate_train(project_name):
    upload_folder = f"./projects/{project_name}/patches/"
    cache_folder = f"./projects/{project_name}/cache/"
    proj = db.session.query(Project).filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400
    current_app.logger.info(f'Generate initial training set for project {project_name}:')

    iset = get_initial_train(cache_folder)
    save_roi = False
    processes = config.getint("pooling","npoolthread", fallback=2)

    #Initial training set selected, start iteration 0
    proj.iteration = 0
    
    rt = multiprocess_insert_patch(iset,proj,project_name,save_roi,processes)
    
    return jsonify(success=True), 200
    #return send_from_directory(upload_folder)

#Multiprocess patch insertion
def multiprocess_insert_patch(iset,proj,project_name,save_roi,processes):
    """iset: tuple (training image set, validation set)"""
    with multiprocessing.Pool(processes=processes) as pool:
        results = [pool.apply_async(insert_patch_into_DB,(proj,project_name,i,save_roi)) for i in iset[0]]
        if not iset[1] is None:
            for v in iset[1]:
                results.append(pool.apply_async(insert_patch_into_DB,(proj,project_name,v,save_roi)) )
        rt = [r.get() for r in results]
    return rt

#Insert patches selected by other means other than user manual upload
def insert_patch_into_DB(proj,project_name,img,save_roi):
    #Make tile for patch
    tilename = None
    pdest = ""
    filename = os.path.basename(img.getPath())
    tile,x,y,ps = tile_for_patch(f"./projects/{project_name}/{filename}")
    newImage = None
    
    if tile is None:
        current_app.logger.info(f'Project = {str(proj.id)}: No WSI directory available')
        dest = f"./projects/{project_name}/{filename}"
        # Check if the file name has been used before
        if os.path.isfile(dest):
            return jsonify(error="file already exists"), 400
        shutil.copy(img.getPath(),dest)
        # if it's not a png image
        filebase, fileext = os.path.splitext(filename)
        dim = img.getImgDim()
    else:
        pdest = f"./projects/{project_name}/patches/{filename}"
        tilename = os.path.basename(tile)
        # if it's not a png image
        filebase, fileext = os.path.splitext(tilename)
        dest = f"./projects/{project_name}/tiles/{tilename}"
        if os.path.isfile(dest):
            newImage = db.session.query(Image).filter_by(projId=proj.id, name=tilename).first()
            if not newImage is None:
                print(f"Tile with multiple patches ({tilename})")
                print("Tile in db: {}".format(newImage))
        if os.path.isfile(pdest):
            return jsonify(error="Patch already exists")
        #shutil.copy(img.getPath(),pdest)
        # Get image dimension
        dim = (config.getint("common","tilesize", fallback=2000),)*2
        #Patches are ROIs from the tile
        save_roi = True
        
    current_app.logger.info(f'Destination = {dest}')

    # Save the new image information to database, if it's a new tile
    if newImage is None:
        newImage = Image(name=f"{filebase}.png", path=dest,projId=proj.id,patch_size=ps,
                        width=dim[0], height=dim[1], date=datetime.now())
        db.session.add(newImage)
        db.session.commit()

    if save_roi:
        roi_base_name = f'{filename.replace(".png", "_")}{x}_{y}_roi.png'
        roi_name = f'projects/{project_name}/roi/{roi_base_name}'
        nobjects = db.session.query(Roi).filter_by(imageId=newImage.id).count()
        newRoi = Roi(name=roi_base_name, path=roi_name, alpath=img.getPath(), testingROI = 0, imageId=newImage.id,
                    width=ps, height=ps, x=x, y=y, acq=proj.iteration, nobjects = nobjects,
                    date=datetime.now())
        db.session.add(newRoi)
        db.session.commit()        

    mask_folder = f"projects/{project_name}/mask/"
    mask_name = f"{filebase}.png".replace(".png", "_mask.png")

    mask = PIL.Image.new('RGB', dim)
    mask.save(mask_folder + mask_name, "PNG")

    return jsonify(success=True, image=newImage.as_dict()), 201    

@api.route("/api/<project_name>/get_al_patches", methods=["GET"])
def get_al_patches(project_name):
    proj_folder = f"./projects/{project_name}/"
    proj = db.session.query(Project).filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400
    current_app.logger.info(f'Starting active learning system {project_name}:')

    training_rois = db.session.query(Roi.id, Roi.imageId, Roi.name, Roi.path, Roi.alpath, Roi.testingROI,Roi.height, Roi.width, 
                                Roi.x, Roi.y, Roi.acq, Roi.anclass) \
            .filter(Image.projId == proj.id) \
            .filter(Roi.imageId == Image.id) \
            .group_by(Roi.id).all()

    current_app.logger.info('Retrieved {} annotated training patches (Project {})'.format(len(training_rois),project_name))
    selected = run_al(proj_folder,training_rois,config,proj.iteration)
    if selected is None:
        return jsonify(error="No pool available"),400
    else:
        proj.iteration += 1
        processes = config.getint("pooling","npoolthread", fallback=2)
        current_app.logger.info('Adding selected images to {}: {}'.format(project_name,len(selected)))
        rt = multiprocess_insert_patch((selected,None),proj,project_name,False,processes)
        current_app.logger.info(f'New patches ready to annotation. Sart iteration {proj.iteration}.')
        return jsonify(success=True), 200
    
@api.route("/api/<project_name>/train_autoencoder", methods=["GET"])
def train_autoencoder(project_name):
    proj = db.session.query(Project).filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400
    current_app.logger.info(f'Training autoencoder for project {project_name}:')

    # get the config options:
    current_app.logger.info(f'Getting config options:')
    num_images = config.getint('train_ae', 'numimages', fallback=-1)
    batch_size = config.getint('train_ae', 'batchsize', fallback=32)
    patch_size = config.getint('train_ae', 'patchsize', fallback=256)
    num_workers = config.getint('train_ae', 'numworkers', fallback=0)
    num_epochs = config.getint('train_ae', 'numepochs', fallback=1000)
    num_epochs_earlystop = config.getint('train_ae', 'num_epochs_earlystop', fallback=-1)
    num_min_epochs = config.getint('train_ae', 'num_min_epochs', fallback=300)


    current_app.logger.info(f'Images = {num_images}, epochs = {num_epochs}, batch size = {batch_size}')

    # get the command:
    full_command = [sys.executable,
                    "train_ae.py",
                    f"-n{num_epochs}",
                    f"-p{patch_size}",
                    f"-s{num_epochs_earlystop}",
                    f"-l{num_min_epochs}",
                    f"-m{num_images}",
                    f"-b{batch_size}",
                    f"-r{num_workers}",
                    f"-o./projects/{project_name}/models/0",
                    f"./projects/{project_name}/patches/*.png"]

    current_app.logger.info(full_command)

    # run it asynchronously:
    command_name = "train_autoencoder"

    # Set proj.train_ae_time = null since the model 0 is being retrained, the time should be unavailable
    proj.train_ae_time = None
    db.session.commit()

    return pool_run_script(project_name, command_name, full_command, callback=train_autoencoder_callback)


# This callback updates the train_ae_time value in the database to
# be the amount of time it took for the autoencoder to run:
def train_autoencoder_callback(result):
    # update the job status in the database:
    update_completed_job_status(result)

    # if it was successful, mark the training time in the database:
    retval, jobid = result
    if retval == 0:
        jobs_logger.info('Marking training ae time in database:')
        engine = sqlalchemy.create_engine(get_database_uri())
        projid = engine.connect().execute(f"select projId from job where id = :jobid", jobid=jobid).first()[0]
        engine.connect().execute(
            f"update project set train_ae_time = datetime(), iteration = CASE WHEN iteration<0 then 0 else iteration end where id = :projid",
            projid=projid)
        engine.dispose()


# Fill the training/test files with the available images:
def populate_training_files(project_name, train_file_path, test_file_path):
    # open those text files for writing:
    testfp = open(test_file_path, "w")
    trainfp = open(train_file_path, "w")

    # loop through the images in the database:
    for img in Project.query.filter_by(name=project_name).first().images:  # TODO can improve this
        current_app.logger.info(f'Checking rois for img: {img.name}')
        for roi in img.rois:
            current_app.logger.info(f'Roi path = {roi.path}')
            # check if this image roi exists:
            if not os.path.isfile(roi.path):
                current_app.logger.warn(f'No roi image found at {roi.path}')
                continue

            # append this roi to the appropriate txt file:
            current_app.logger.info(f'Testing ROI = {str(roi.testingROI)}')
            if roi.testingROI:
                testfp.write(f"{roi.name}\n")
            elif roi.testingROI == 0:
                trainfp.write(f"{roi.name}\n")

    # close the files:
    testfp.close()
    trainfp.close()


@api.route("/api/<project_name>/retrain_dl", methods=["GET"])
def retrain_dl(project_name):
    proj = Project.query.filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400
    current_app.logger.info(f'About to train a new transfer model for {project_name}')

    frommodelid = request.args.get('frommodelid', default=0, type=int)

    if(frommodelid == -1):
        frommodelid = get_latest_modelid(project_name)

    if frommodelid > proj.iteration or not os.path.exists(f"./projects/{project_name}/models/{frommodelid}/best_model.pth"):
        return jsonify(
            error=f"Deep learning model {frommodelid} doesn't exist"), 400

    if proj.train_ae_time is None and frommodelid == 0:
        error_message = f'The base model 0 of project {project_name} was overwritten when Retrain Model 0 started.\n ' \
                        f'Please wait until the Retrain Model 0 finishes. '
        current_app.logger.warn(error_message)
        return jsonify(error=error_message), 400

    # todo: make sure there's actually a model in that subdirectory since errors still create the dir before the model is ready
    new_modelid = get_latest_modelid(project_name) + 1
    output_model_path = f"./projects/{project_name}/models/{new_modelid}/"
    current_app.logger.info(f'New model path = {output_model_path}')

    # store the list of test and training images in text files:
    test_file_path = f"projects/{project_name}/test_imgs.txt"
    train_file_path = f"projects/{project_name}/train_imgs.txt"

    current_app.logger.info('Populating project files:')
    populate_training_files(project_name, train_file_path, test_file_path)

    # check if enough data exists:
    empty_training = not os.path.exists(test_file_path) or os.stat(
        test_file_path).st_size == 0
    empty_testing = not os.path.exists(test_file_path) or os.stat(
        test_file_path).st_size == 0
    if empty_training or empty_testing:  # TODO can improve this by simply counting ROIs in the db
        error_message = f'Not enough training/test images for project {project_name}. You need at least 1 of each.'
        current_app.logger.warn(error_message)
        return jsonify(error=error_message), 400

    # get config properties:
    num_epochs = config.getint('train_tl', 'numepochs', fallback=1000)
    num_epochs_earlystop = config.getint('train_tl', 'num_epochs_earlystop', fallback=-1)
    num_min_epochs = config.getint('train_tl', 'num_min_epochs', fallback=300)
    batch_size = config.getint('train_tl', 'batchsize', fallback=32)
    patch_size = config.getint('train_tl', 'patchsize', fallback=256)
    num_workers = config.getint('train_tl', 'numworkers', fallback=0)
    edge_weight = config.getfloat('train_tl', 'edgeweight', fallback=2)
    pclass_weight = config.getfloat('train_tl', 'pclass_weight', fallback=.5)
    fillbatch = config.getboolean('train_tl', 'fillbatch', fallback=False)

    # query P/N pixel count from database for ppixel_train npixel_train ppixel_test npixel_test
    if pclass_weight == -1:
        proj_ppixel = db.session.query(db.func.sum(Image.ppixel)).filter_by(
            projId=proj.id).scalar()
        proj_npixel = db.session.query(db.func.sum(Image.npixel)).filter_by(
            projId=proj.id).scalar()
        total = proj_npixel + proj_ppixel
        pclass_weight = 1 - proj_ppixel / total

    # get the command to retrain the model:
    full_command = [sys.executable, "train_model.py",
                    f"-p{patch_size}",
                    f"-e{edge_weight}",
                    f"-n{num_epochs}",
                    f"-s{num_epochs_earlystop}",
                    f"-l{num_min_epochs}",
                    f"-b{batch_size}",
                    f"-o{output_model_path}",
                    f"-w{pclass_weight}",
                    f"-r{num_workers}",
                    f"-m./projects/{project_name}/models/{frommodelid}/best_model.pth",
                    f"./projects/{project_name}"]

    if(fillbatch):
        full_command.append("--fillbatch")

    current_app.logger.info(f'Training command = {full_command}')

    # run the script asynchronously:
    command_name = "retrain_dl"
    return pool_run_script(project_name, command_name, full_command, callback=retrain_dl_callback)


def retrain_dl_callback(result):
    # update the job status in the database:
    update_completed_job_status(result)

    jobid = result[1]
    engine = sqlalchemy.create_engine(get_database_uri())

    dbretval = engine.connect().execute(f"select procout from jobid_{jobid} where procout like 'RETVAL:%'").first()
    if dbretval is None:
        # no retval, indicating superpixel didn't get to the end, leave everything as is
        engine.dispose()
        return

    retvaldict = json.loads(dbretval[0].replace("RETVAL: ", ""))
    projname = retvaldict["project_name"]
    iteration = retvaldict["iteration"]

    engine.connect().execute(f"update project set iteration  = :iteration where name = :projname",
                             projname=projname, iteration=iteration)
    engine.dispose()


@api.route("/api/<project_name>/make_patches", methods=["GET"])
def make_patches(project_name):
    # pull this project from the database:
    current_app.logger.info(f'Getting project info from database for project {project_name}.')
    project = db.session.query(Project).filter_by(name=project_name).first()
    if project is None:
        current_app.logger.warn(f'Unable to find {project_name} in database. Returning HTML response code 400.')
        return jsonify(error=f"Project {project_name} does not exist"), 400

    target_files = []
    current_app.logger.info('Looping through images.')
    for img in project.images:
        current_app.logger.info(f'Checking database if patches have been computed for image "{img.name}".')
        needs_calculating = False
        if img.make_patches_time:
            current_app.logger.info('Database claims that the patches have been computed. Checking filesystem.')
            image_name_without_extension = os.path.splitext(img.name)[0]  # <-- remove extension
            current_app.logger.info(f'Image {image_name_without_extension}')
            patches_pattern = f'./projects/{project_name}/patches/{image_name_without_extension}*.png'
            current_app.logger.info(f'Patches pattern = {patches_pattern}')
            number_of_patches = len(glob.glob(patches_pattern))
            current_app.logger.info(f'Number of patches = {number_of_patches}')
            if number_of_patches == 0:
                current_app.logger.warn(
                    'The database is incorrectly reporting that patches exist. We are recomputing them since no patches exist on the filesystem for this image.')
                needs_calculating = True
        else:
            needs_calculating = True

        if needs_calculating:
            current_app.logger.info(
                f'Patches need to be computed for image at {img.path}. Adding this image to the list.')
            target_files.append(img.path)
            #            img.patches_computed = True  # note, this only goes through when commit is called
            current_app.logger.info('Marked patches_computed to be True in the database.')

    if not target_files:
        error_message = 'No pending target image files for making patches.'
        current_app.logger.warn(error_message)
        return jsonify(error=error_message), 400

    current_app.logger.info('Storing image filenames for patches in text file:')
    with open(f"./projects/{project_name}/patches/new_imgs.txt", "w") as textfile:
        for fname in target_files:
            textfile.write(f"{fname}\n")

    patchsize = config.getint('make_patches', 'patchsize', fallback=256)

    # get the command:
    full_command = [sys.executable,
                    "make_patches_for_embed.py", f"-p{patchsize}",
                    f"-o./projects/{project_name}/patches/",
                    f"./projects/{project_name}/patches/new_imgs.txt"]

    whiteBG = request.args.get("whiteBG", default="keep", type=str)
    if whiteBG == "remove":
        full_command.append("-b")

    current_app.logger.info(full_command)

    # close the db session and note that patches_computed is true:
    db.session.commit()

    # run the command asynchronously
    command_name = "make_patches"
    return pool_run_script(project_name, command_name, full_command, callback=make_patches_callback)


def make_patches_callback(result):
    # update the job status in the database:
    update_completed_job_status(result)

    retval, jobid = result
    engine = sqlalchemy.create_engine(get_database_uri())
    dbretval = engine.connect().execute(f"select procout from jobid_{jobid} where procout like 'RETVAL:%'").first()
    if dbretval is None:
        # no retval, indicating make_patches didn't get to the end, leave everything as is
        engine.dispose()
        return

    retvaldict = json.loads(dbretval[0].replace("RETVAL: ", ""))
    for img in retvaldict["image_list"]:
        engine.connect().execute(f"update image set make_patches_time = datetime() where path= :img", img=img)

    # if it was successful, mark the training time in the database:

    if retval == 0:
        jobs_logger.info('Marking make_patches time in database:')
        projid = engine.connect().execute(f"select projId from job where id = :jobid", jobid=jobid).first()[0]
        engine.connect().execute(f"update project set make_patches_time = datetime() where id = :projid", projid=projid)

    engine.dispose()


@api.route("/api/<project_name>/embed", methods=["GET"])
def make_embed(project_name):
    proj = db.session.query(Project).filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400

    model0ExistOrNot = os.path.exists(f"./projects/{project_name}/models/0/best_model.pth")
    current_app.logger.info(f'Model 0 (autoencoder) exists = {model0ExistOrNot}')
    if not model0ExistOrNot:
        return jsonify(
            error="Embedding is not available unless at least a base model is trained. Please make patches and train AE"), 400

    if proj.train_ae_time is None and proj.iteration == 0:
        error_message = f'The base model 0 of project {project_name} was overwritten when Retrain Model 0 started.\n ' \
                        f'Please wait until the Retrain Model 0 finishes. '
        current_app.logger.warn(error_message)
        return jsonify(error=error_message), 400

    current_app.logger.info('Checking if the embeddings are the most recent.')

    # get config options:
    batchsize = config.getint('make_embed', 'batchsize', fallback=32)
    patchsize = config.getint('make_embed', 'patchsize', fallback=256)
    numimgs = request.args.get('numimgs', default=-1, type=int)
    modelid = request.args.get('modelid', default=get_latest_modelid(project_name), type=int)
    outdir = f"./projects/{project_name}/models/{modelid}"

    latest_modelID = get_latest_modelid(project_name)

    if modelid < 0 or modelid > latest_modelID:
        return jsonify(
            error=f"Your selected Embed Model ID is {modelid}. The last model ID is {latest_modelID}. A valid Model ID ranges from 0 to {latest_modelID}."), 400

    # get the command:
    full_command = [sys.executable, "make_embed.py", project_name, f"-o{outdir}", f"-p{patchsize}", f"-b{batchsize}",
                    f"-m{numimgs}"]
    current_app.logger.info(f'Full command = {str(full_command)}')

    # update the embedding iteration:
    # current_app.logger.info('Updating the embedding iteration to the model iteration:')
    # proj.embed_iteration = proj.iteration
    db.session.commit()

    # run the command asynchronously:
    command_name = "make_embed"
    return pool_run_script(project_name, command_name, full_command, callback=make_embed_callback)


def make_embed_callback(result):
    # update the job status in the database:
    update_completed_job_status(result)

    jobid = result[1]
    engine = sqlalchemy.create_engine(get_database_uri())

    dbretval = engine.connect().execute(f"select procout from jobid_{jobid} where procout like 'RETVAL:%'").first()
    if dbretval is None:
        # no retval, indicating superpixel didn't get to the end, leave everything as is
        engine.dispose()
        return

    retvaldict = json.loads(dbretval[0].replace("RETVAL: ", ""))
    projname = retvaldict["project_name"]
    modelid = retvaldict["modelid"]

    engine.connect().execute(f"update project set embed_iteration = :modelid where name = :projname", projname=projname,
                             modelid=modelid)

    engine.dispose()


@api.route("/api/<project_name>/model", methods=["GET"])
def get_model(project_name):
    modelid = request.args.get('model', get_latest_modelid(project_name), type=int)
    model_path = f"./projects/{project_name}/models/{modelid}/"
    return send_from_directory(model_path, "best_model.pth", as_attachment=True)


@api.route('/api/<project_name>/dataset/<traintype>', methods=["GET"])
def get_traintest_images(project_name, traintype):
    # List all training and testing patches for the current project
    sample_images = []
    with open(f"projects/{project_name}/{traintype}_imgs.txt", "r") as file:
        for img_name in file:
            sample_images.append(img_name.strip())

    current_app.logger.info(sample_images)
    return jsonify(sample_images=sample_images)


@api.route('/api/<project_name>/dataset/<traintype>/<roiname>', methods=["DELETE"])
def remove_image_from_traintest(project_name, traintype, roiname):
    roi = db.session.query(Roi).filter(name=os.path.basename(roiname.strip())).first()
    roi.testingROI = -1
    db.session.commit()

    return jsonify(success=True, roi=roi.as_dict())


@api.route('/api/<project_name>/dataset/<traintype>/<roiname>/<roi_class>', methods=["PUT"])
def add_roi_to_traintest(project_name, traintype, roiname, roi_class):
    current_app.logger.info(
        f'Adding new annotation image. Project = {project_name} Training type = {traintype} Name = {roiname}')
    
    roi = db.session.query(Roi).filter_by(name=os.path.basename(roiname.strip())).first()
    if roi is None:
        return jsonify(error=f"{roiname} not found in project {project_name}"), 400
    current_app.logger.info('Roi found = ' + str(roi.id))

    if traintype == "train":
        roi.testingROI = 0
    if traintype == "test":
        roi.testingROI = 1

    roi.anclass = 1 if roi_class == "positive" else 0

    current_app.logger.info('Committing new image to database:')
    db.session.commit()

    return jsonify(success=True, roi=roi.as_dict()), 200


@api.route("/api/<project_name>/image/<image_name>", methods=["GET"])
def get_image(project_name, image_name):
    current_app.logger.info(f"Outputting file {image_name}")
    return send_from_directory(f"./projects/{project_name}/tiles", image_name)

@api.route("/api/<project_name>/image/<image_name>/thumbnail", methods=["GET"])
def get_image_thumb(project_name, image_name):
    width = request.form.get('width', 250)

    img = cv2.imread(f"./projects/{project_name}/tiles/{image_name}")

    height = int(img.shape[0] * width / img.shape[1])
    dim = (width, height)
    img = cv2.resize(img, dim)

    success, img_encoded = cv2.imencode('.png', img)

    response = make_response(img_encoded.tobytes())
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Disposition'] = f'inline; filename = "{image_name.replace(".png", "_thumb.png")}"'

    return response


@api.route('/api/<project_name>/image/<image_name>',
           methods=["DELETE"])  # below should be done in a post-processing call
def delete_image(project_name, image_name):
    proj = Project.query.filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400

    # Remove the image from database
    selected_image = db.session.query(Image).filter_by(projId=proj.id, name=image_name).first()

    # Delete all the ROI linked to the image
    select_Rois = db.session.query(Roi).filter_by(imageId=selected_image.id)
    select_Rois.delete()

    db.session.delete(selected_image)
    db.session.commit()

    # Remove the image file from server
    os.remove(selected_image.path)

    # Remove the corresponding mask and result files
    # TODO: the below can be refactored to recursively look for *all* files which match the pattern and delete them
    # need to be careful with recursive search. if one has 1.png and 100.png, and wants to delete all files associated
    # with 1.png, wildcards may pick up unrelated 100.png images
    mask_name = selected_image.name.replace(".png", "_mask.png")
    mask_path = f"./projects/{project_name}/mask/{mask_name}"
    if os.path.exists(mask_path):
        os.remove(mask_path)

    # --- delete prediction results for every model
    result_name = selected_image.name.replace(".png", "_pred.png")
    result_path = f"./projects/{project_name}/pred/**/{result_name}"
    result_fileLists = glob.glob(result_path)
    for filePath in result_fileLists:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    # --- delete patches
    patches = selected_image.name.replace(".png", "")
    patches_path = f"./projects/{project_name}/patches"
    patches_fileList = glob.glob(f'{patches_path}/{patches}_*_*.png')

    # Iterate over the list of filepaths & remove each file.
    for filePath in patches_fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    # --- delete superpixels
    superpixels_name = selected_image.name.replace(".png", "_superpixels.png")
    superpixels_path = f"./projects/{project_name}/superpixels/{superpixels_name}"
    if os.path.exists(superpixels_path):
        os.remove(superpixels_path)

    superpixels_boundary_name = selected_image.name.replace(".png", "_superpixels_boundary.png")
    superpixels_boundary_path = f"./projects/{project_name}/superpixels_boundary/{superpixels_boundary_name}"
    if os.path.exists(superpixels_boundary_path):
        os.remove(superpixels_boundary_path)

    # Todo: Remove the image patches from embedding

    # Get the image list for the project
    return jsonify(success=True), 204


@api.route("/api/<project_name>/image", methods=["POST"])
def upload_image(project_name):
    current_app.logger.info(f'Uploading image for project {project_name} :')

    # ---- check project exists first!
    proj = Project.query.filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400
    current_app.logger.info(f'Project = {str(proj.id)}')

    file = request.files.get('file')
    filename = file.filename
    save_roi = False
    pdest = f"./projects/{project_name}/patches/{filename}"
    file.save(pdest)
    
    img = get_img_metadata(pdest)
    ret = insert_patch_into_DB(proj,project_name,img,save_roi)

    return ret

@api.route("/api/<project_name>/roi/<roi_name>/mask", methods=["GET"])
def get_roimask(project_name, roi_name):
    mask_folder = f"projects/{project_name}/mask/"
    match = re.search(r"(.*)_(\d+)_(\d+)_roi.png", roi_name)
    mask_name = f"{match.group(1)}_mask.png"
    x = int(match.group(2))
    y = int(match.group(3))

    roi = cv2.imread(f"./projects/{project_name}/roi/{roi_name}")
    if roi is None:
        jsonify(error=f"ROI file {roi_name} does not exist"), 400

    h = roi.shape[0]
    w = roi.shape[1]

    mask = cv2.imread(mask_folder + mask_name)
    mask = mask[y:y + h, x:x + w, :]

    success, mask_encoded = cv2.imencode('.png', mask)

    response = make_response(mask_encoded.tobytes())
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Disposition'] = f'inline; filename = "{roi_name.replace(".png", "_mask.png")}"'

    return response


@api.route("/api/<project_name>/image/<image_name>/roimask", methods=["POST"])
def post_roimask(project_name, image_name):
    current_app.logger.info(f'Uploading roi mask for project {project_name} and image {image_name}:')
    proj = Project.query.filter_by(name=project_name).first()
    if proj is None:
        return jsonify(error=f"project {project_name} doesn't exist"), 400

    current_app.logger.info(f'Project id = {str(proj.id)}')
    force = request.form.get('force', False, type=bool)

    selected_image = db.session.query(Image).filter_by(projId=proj.id,
                                                       name=image_name).first()
    if selected_image is None:
        return jsonify(error=f"{selected_image} inside of project {project_name} doesn't exist"), 400

    roimask_url = request.form.get('roimask', None)

    if not roimask_url:
        return jsonify(error="no roimask provided"), 400

    roimask_data = re.search(r'data:image/png;base64,(.*)', roimask_url).group(1)
    roimask_decoded = base64.b64decode(roimask_data)
    roimask = cv2.imdecode(np.frombuffer(roimask_decoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    roimask = cv2.cvtColor(roimask, cv2.COLOR_BGR2RGB)

    if not np.all(np.isin(roimask, [0, 255])):
        return jsonify(error="Non [0,255] incorrect values are saved in the roimask mask, please check"), 400

    if roimask.shape[2] > 3:
        return jsonify(error="Roi Mask has 4 dimensions? Possible Alpha Channel Issue?"), 400

    h = roimask.shape[0]
    w = roimask.shape[1]

    x = int(request.form.get('pointx', -1))
    y = int(request.form.get('pointy', -1))

    if -1 == x or -1 == y:
        return jsonify(error="no x , y location provided"), 402

    img = cv2.imread(f"./projects/{project_name}/tiles/{image_name}")
    if y + h > img.shape[0] or x + w > img.shape[1] or y < 0 or x < 0:
        return jsonify(f"ROI not within image, roi xy ({x} ,{y}) vs image size ({img.shape[0]}, {img.shape[1]})"), 400



    mask_name = f"projects/{project_name}/mask/{image_name.replace('.png', '_mask.png')}"
    if not os.path.isfile(mask_name):
        mask = np.zeros(img.shape, dtype=np.uint8)
    else:
        mask = cv2.cvtColor(cv2.imread(mask_name), cv2.COLOR_BGR2RGB)

    roimaskold = mask[y:y + h, x:x + w, :]

    if not force and np.any(roimaskold != 0):
        current_app.logger.error('ROI exists at this position.')
        return jsonify(error="ROI at this position already exists, enable force to overide"), 402

    mask[y:y + h, x:x + w, :] = roimask
    cv2.imwrite(mask_name, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    roi_base_name = f'{image_name.replace(".png", "_")}{x}_{y}_roi.png'
    roi_name = f'projects/{project_name}/roi/{roi_base_name}'

    roi = img[y:y + h, x:x + w, :]
    cv2.imwrite(roi_name, roi)

    # --- update positive / negative stats

    selected_image.ppixel = np.count_nonzero(mask[:, :, 1] == 255)
    selected_image.npixel = np.count_nonzero(mask[:, :, 0] == 255)
    
    # -- determine number of new objects from this roi, will need for statistics later
    nobjects_roi = get_number_of_objects(roimask)
    selected_image.nobjects = get_number_of_objects(mask)

    # ----
    parent_image = Image.query.filter_by(name=image_name, projId=proj.id).first()
    rois = Roi.query.filter_by(imageId=parent_image.id)
    newRoi = None
    for r in rois:
        if abs(r.x-x) <= 2 and abs(r.y - y) <= 2:
            newRoi = r
        else:
            current_app.logger.info("Found ROI ({}) with different coordinates: ({},{}) against ({},{})".format(r.id,r.x,r.y,x,y))
            
    if newRoi:
        current_app.logger.info("ROI already stored. Original patch: {}".format(newRoi.alpath))
        newRoi.nobjects = nobjects_roi
        newRoi.path = roi_name
        db.session.commit()
    else:
        current_app.logger.info('Storing roi to database:')
        
        newRoi = Roi(name=roi_base_name, path=roi_name, imageId=parent_image.id,
                    width=w, height=h, x=x, y=y, nobjects = nobjects_roi,
                    date=datetime.now())
        db.session.add(newRoi)
        db.session.commit()

    return jsonify(success=True, roi=newRoi.as_dict()), 201


@api.route("/api/<project_name>/roi/<roi_name>", methods=["GET"])
def get_roi(project_name, roi_name):
    response = send_from_directory(f"./projects/{project_name}/roi/", roi_name)

    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@api.route("/api/<project_name>/image/<image_name>/mask", methods=["GET"])
def get_mask(project_name, image_name):
    response = send_from_directory(f"./projects/{project_name}/mask",
                                   image_name.replace(".png", "_mask.png"))

    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@api.route("/api/<project_name>/image/<image_name>/prediction", methods=["GET"])
def get_prediction(project_name, image_name):
    current_app.logger.info(f'Getting prediction for project {project_name} and image {image_name}')

    project = Project.query.filter_by(name=project_name).first()
    curr_image = Image.query.filter_by(projId=project.id, name=image_name).first()
    if curr_image is None:
        jsonify(error=f"Image {image_name} does not exist"), 400

    modelid = request.args.get('model', get_latest_modelid(project_name), type=int)
    current_app.logger.info(f'Model id = {str(modelid)}')

    if modelid <= 0:
        current_app.logger.warn(f"No DL model trained for {project_name} -- {image_name} -- {modelid}")
        return jsonify(error="No AI model trained, so no AI results available yet."), 400

    upload_folder = f"./projects/{project_name}/pred/{modelid}"
    fname = image_name.replace(".png", "_pred.png")
    full_fname = f"{upload_folder}/{fname}"
    current_app.logger.info('Full filename for prediction = ' + full_fname)

    print('Generating new prediction image:')
    batchsize = config.getint('get_prediction', 'batchsize', fallback=32)
    patchsize = config.getint('get_prediction', 'patchsize', fallback=256)

    # run the command:
    full_command = [sys.executable, "make_output_unet_cmd.py", f"-s{batchsize}", f"-p{patchsize}",
                    f"-m./projects/{project_name}/models/{modelid}/best_model.pth",
                    f"-o./projects/{project_name}/pred/{modelid}",
                    f"./projects/{project_name}/{image_name}", "--force"]

    command_name = "generate_prediction"
    return pool_get_image(project_name, command_name, full_command, full_fname, imageid=curr_image.id)


@api.route("/api/<project_name>/image/<image_name>/superpixels", methods=["GET"])
def get_superpixels(project_name, image_name):
    current_app.logger.info(f'Getting superpixel for project {project_name} and image {image_name}')
    latest_modelid = get_latest_modelid(project_name)

    force = request.args.get('force', False, type=bool)

    modelidreq = request.args.get('superpixel_run_id', latest_modelid, type=int)
    current_app.logger.info(f'Model id = {str(modelidreq)}')
    if modelidreq > latest_modelid:
        return jsonify(error=f"Requested ModelID {modelidreq} greater than available models {latest_modelid}"), 400

    project = Project.query.filter_by(name=project_name).first()
    curr_image = Image.query.filter_by(projId=project.id, name=image_name).first()
    superpixel_modelid = curr_image.superpixel_modelid
    current_app.logger.info(f'The current superpixel_modelid of {image_name} = {str(superpixel_modelid)}')

    upload_folder = f"./projects/{project_name}/superpixels"
    spixel_fname = image_name.replace(".png", "_superpixels.png")
    full_fname = f"{upload_folder}/{spixel_fname}"
    current_app.logger.info('Full filename for superpixel = ' + full_fname)

    batchsize = config.getint('superpixel', 'batchsize', fallback=32)
    patchsize = config.getint('superpixel', 'patchsize', fallback=256)
    approxcellsize = config.getint('superpixel', 'approxcellsize', fallback=20)
    compactness = config.getfloat('superpixel', 'compactness', fallback=.01)
    command_to_use = config.get("superpixel", 'command_to_use', fallback="make_superpixel.py")

    if modelidreq < 0:
        # We are using simple method, since we have no dl model
        current_app.logger.warn(
            f"No DL model trained for {project_name} -- {image_name} -- {modelidreq}, will use simple method")
        command_to_use = "make_superpixel.py"

    full_command = [sys.executable, command_to_use,
                    f"-p{patchsize}",
                    f"-x{batchsize}",
                    f"-c{compactness}",
                    f"-a{approxcellsize}",
                    f"-m./projects/{project_name}/models/{modelidreq}/best_model.pth",
                    f"-s./projects/{project_name}/superpixels/",
                    f"-o./projects/{project_name}/superpixels_boundary/",
                    f"./projects/{project_name}/{image_name}", "--force"]

    current_app.logger.info(
        f'We are running {command_to_use} to generate superpixels for IMAGE {image_name} in PROJECT {project_name} ')
    current_app.logger.info(f'Superpixel command = {full_command}')

    command_name = "generate_superpixel"

    if modelidreq > superpixel_modelid or force:
        try:
            os.remove(full_fname)
        except:
            pass

    return pool_get_image(project_name, command_name, full_command, full_fname, imageid=curr_image.id,
                          callback=get_superpixels_callback)


def get_superpixels_callback(result):
    # update the job status in the database:
    update_completed_job_status(result)

    retval, jobid = result
    engine = sqlalchemy.create_engine(get_database_uri())

    dbretval = engine.connect().execute(f"select procout from jobid_{jobid} where procout like 'RETVAL:%'").first()
    if dbretval is None:
        # no retval, indicating superpixel didn't get to the end, leave everything as is
        engine.dispose()
        return

    retvaldict = json.loads(dbretval[0].replace("RETVAL: ", ""))
    if "model" in retvaldict:  # for DL approach
        modelid = retvaldict["model"].split("/")[4]
    else:
        modelid = -1

    for img in retvaldict["output_file"]:
        engine.connect().execute(
            f"update image set superpixel_time = datetime(), superpixel_modelid = :modelid where path= :img", img=img,
            modelid=modelid)

    engine.dispose()


@api.route("/api/<project_name>/image/<image_name>/superpixels_boundary", methods=["GET"])
def get_superpixels_boundary(project_name, image_name):
    upload_folder = f"./projects/{project_name}/superpixels_boundary"
    spixel_fname = image_name.replace(".png", "_superpixels_boundary.png")
    full_fname = f"{upload_folder}/{spixel_fname}"

    oseg_fname = f'./projects/{project_name}/superpixels/{image_name.replace(".png", "_superpixels.png")}'

    if not os.path.isfile(oseg_fname):
        return jsonify(error="need to generate superpixels image first"), 400

    folder, filename = os.path.split(full_fname)
    response = send_from_directory(folder, filename)

    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@api.route("/api/<project_name>/image/<image_name>/<direction>", methods=["GET"])
def prevnext_image(project_name, image_name, direction):
    project = Project.query.filter_by(name=project_name).first()
    curr_image = Image.query.filter_by(projId=project.id, name=image_name).first()

    # To do: we can not prev the "first image" and "next" the last image, need to make it periodic
    if (direction == "previous"):
        image = Image.query.filter((Image.id < curr_image.id) & (Image.projId == project.id)) \
            .order_by(Image.id.desc()).first()
    else:
        image = Image.query.filter((Image.id > curr_image.id) & (Image.projId == project.id)) \
            .order_by(Image.id.asc()).first()

    current_app.logger.info(f"{project_name} -- {image_name} --- {direction}")

    if image is None:
        errorMessage = "There is no " + direction + " image"
        return jsonify(error=errorMessage), 400
    else:
        return jsonify(url=url_for('html.annotation', project_name=project_name, image_name=image.name)), 200


# ---- config work
@api.route('/api/config', methods=["GET"])
def getconfig():  # Front end can now keep track of the last lines sent and request all the "new" stuff
    allsections = dict()
    for section in config.sections():
        sectionitems = []
        for items in config[section].items():
            sectionitems.append(items)
        allsections[section] = sectionitems
    return jsonify(allsections)


@api.route("/api/<project_name>/embedcsv", methods=["GET"])
def get_embed_csv(project_name):
    project = Project.query.filter_by(name=project_name).first()

    latest_modelid = get_latest_modelid(project_name)
    selected_modelid = request.args.get('modelid', default=latest_modelid, type=int)
    fname = f"./projects/{project_name}/models/{selected_modelid}/embedding.csv"

    if selected_modelid > latest_modelid or selected_modelid < 0:
        error_message = f"Your selected View Embed Model ID is {selected_modelid}. A valid Model ID ranges from 0 to {latest_modelid}."
        current_app.logger.error(error_message)
        return jsonify(
            error=error_message), 400

    if not os.path.exists(fname):
        error_message = f'No embedding data available to render for Model {selected_modelid}.'
        current_app.logger.error(error_message)
        return jsonify(
            error=error_message), 400

    folder, filename = os.path.split(fname)
    response = send_from_directory(folder, filename)

    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response



def get_number_of_objects(img):
    _, nobjects = label(img[:, :, 1], return_num=True)
    return nobjects



