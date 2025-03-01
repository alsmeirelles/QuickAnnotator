from flask import render_template, Blueprint, request, current_app
from flask_sqlalchemy import SQLAlchemy

# from QA_api import get_traintest_images
import QA_api
from QA_config import config
from QA_db import Image, Project, Job, Roi, get_latest_modelid

html = Blueprint("html", __name__, static_folder="static", template_folder="templates")

db = SQLAlchemy()


@html.route('/favicon.ico')
def favicon():
    return html.send_static_file("favicon.ico")


@html.route('/')
def index():
    projects = db.session.query(Project.name, Project.date, Project.iteration, Project.description, Project.id,
                                Project.images,
                                db.func.count(Roi.id).label('nROIs'),
                                (db.func.count(Roi.id) - db.func.ifnull(db.func.sum(Roi.testingROI), 0))
                                .label('nTrainingROIs'), db.func.count(db.func.distinct(Image.id)).label('nImages'),
                                db.func.ifnull(db.func.sum(db.func.distinct(Image.nobjects)), 0).label('nObjects')) \
        .outerjoin(Image, Image.projId == Project.id) \
        .outerjoin(Roi, Roi.imageId == Image.id).group_by(Project.id).all()

    return render_template("index.html", projects=projects)


@html.route('/<project_name>', methods=['GET'])
@html.route('/<project_name>/images', methods=['GET'])
@html.route('/<project_name>/testimages/<testing>', methods=['GET'])
def get_imagelist(project_name,testing=0):
    # Get the image list for the project
    project = Project.query.filter_by(name=project_name).first()

    if not project:
        return render_template("error.html")
    
    testing = int(testing)
    if testing == 0:
        images = db.session.query(Image.id, Image.projId, Image.name, Image.path, Image.height, Image.width, Image.date,
                                Image.rois, Image.make_patches_time, Image.npixel, Image.ppixel, Image.nobjects,
                                db.func.count(Roi.id).label('ROIs'),
                                (db.func.count(Roi.id) - db.func.ifnull(db.func.sum(Roi.testingROI), 0))
                                .label('trainingROIs')). \
            outerjoin(Roi, Roi.imageId == Image.id). \
            filter(Image.projId == project.id).filter(Roi.acq == project.iteration).filter(Roi.testingROI == 0).group_by(Image.id).all()
    else:
        images = db.session.query(Image.id, Image.projId, Image.name, Image.path, Image.height, Image.width, Image.date,
                                Image.rois, Image.make_patches_time, Image.npixel, Image.ppixel, Image.nobjects,
                                db.func.count(Roi.id).label('ROIs'),
                                (db.func.count(Roi.id) - db.func.ifnull(db.func.sum(Roi.testingROI), 0))
                                .label('trainingROIs')). \
            outerjoin(Roi, Roi.imageId == Image.id). \
            filter(Image.projId == project.id).filter(Roi.testingROI == 1).filter(Roi.anclass == -1).group_by(Image.id).all()
        current_app.logger.info('Displaying TESTING ROIs: {}'.format(len(images)))
        
    return render_template("images.html", project=project, images=images,testing=testing)


@html.route('/<project_name>/images/images-main', methods=['GET'])
def images_main(project_name):
    # Get the image list for the project
    project = Project.query.filter_by(name=project_name).first()
    if not project:
        return render_template("error.html")
    else:
        return render_template("images-main.js", project=project)


@html.route('/<project_name>/dataset/<type>', methods=['GET'])
def display_sample_images(project_name, type):
    project = Project.query.filter_by(name=project_name).first()

    if not project:
        return render_template("error.html")

    imglist = QA_api.get_traintest_images(project_name, type)
    return render_template("sampleimages.html", project_name=project_name, imglist=imglist)


@html.route("/<project_name>/embed", methods=['GET'])
def plotembed(project_name):
    current_app.logger.info('Plotting patch embedding:')
    project = Project.query.filter_by(name=project_name).first()
    if not project:
        current_app.logger.error('No project found.')
        return render_template("error.html")

    latest_modelid = get_latest_modelid(project_name)
    selected_modelid = request.args.get('modelid', default=latest_modelid, type=int)
    if selected_modelid > latest_modelid or selected_modelid < 0:
        error_message = f"Your selected View Embed Model ID is {selected_modelid}. A valid Model ID ranges from 0 to {latest_modelid}."
        current_app.logger.error(error_message)
        return render_template("embed.html", project_name=project_name, data="None",
                               project_iteration=project.iteration, current_modelId=selected_modelid,
                               error_message=error_message)

    return render_template("embed.html", project_name=project_name, project_iteration=project.iteration,
                           current_modelId=selected_modelid)


@html.route("/<project_name>/embed/embed-main.js", methods=['GET'])  # --- should not need this function
def embed_main(project_name):
    # Get the image list for the project
    project = Project.query.filter_by(name=project_name).first()
    if not project:
        return render_template("error.html")

    return render_template("embed-main.js", project_name=project_name)


@html.route('/<project_name>/<image_id>/annotation', methods=['GET'])
@html.route('/<project_name>/<image_id>/<testing>/annotation', methods=['GET'])
def annotation(project_name, image_id, testing=0):
    project = Project.query.filter_by(name=project_name).first()

    if not project:
        return render_template("error.html")

    # Method 1
    # image = Image.query.filter_by(projId=project.id, name=image_name).first()
    # image.nROIs = Roi.query.filter_by(imageId=image.id).count()
    # image.nTrainingROIs = Roi.query.filter_by(imageId=image.id, testingROI=0).count()

    # Method 2 (corresponding sql code)
    # SELECT image.id, count(roi.id)
    # FROM image
    # JOIN roi
    # ON roi.imageId = image.id
    # WHERE image.id = 1
    # GROUP BY image.id

    image = db.session.query(Image.id, Image.projId, Image.name, Image.path, Image.patch_size, Image.height, Image.width, Image.date,
                             Image.rois, Image.make_patches_time, Image.nobjects,
                             db.func.count(Roi.id).label('nROIs'),
                             (db.func.count(Roi.id) - db.func.ifnull(db.func.sum(Roi.testingROI), 0))
                             .label('nTrainingROIs')). \
        outerjoin(Roi, Roi.imageId == Image.id). \
        filter(Image.id == image_id).filter(Image.projId == project.id).group_by(Image.id).first()

    x = request.args.get('startX', "#")
    y = request.args.get('startY', "#")
    defaultCropSize = config.getint('common', 'patchsize', fallback=256)

    if image.nROIs > 0:
        if not testing:
            rois = db.session.query(Roi,Roi.alpath,Roi.x,Roi.y).filter_by(projId=project.id,imageId=image.id,acq=project.iteration).all()
        else:
            rois = db.session.query(Roi,Roi.alpath,Roi.x,Roi.y).filter_by(projId=project.id,imageId=image.id).all()
        roi = rois[0]
        x,y = roi.x,roi.y
        forceCropSize = image.patch_size
        print("Rois in tile (id {}):\n {}".format(image_id,"\n".join(["- X {}; Y {}".format(r.x,r.y) for r in rois])))
        print("Image is a tile for patch: {}. Start (x,y): {}. Size: {}".format(roi.alpath,(x,y),defaultCropSize))
    else:
        print("Image has no ROIs for this iteration: {}".format(image_name))
        
    return render_template("annotation.html", project=project, image=image, startX=x, startY=y,
                           defaultCropSize=defaultCropSize,forceCropSize=forceCropSize,testing=testing)


# For templates which just use the project and image name:
def rendered_project_image(template_name, project_name, image_name,image_id):
    project = Project.query.filter_by(name=project_name).first()
    image = Image.query.filter_by(projId=project.id, id=image_id).first()
    defaultCropSize = config.getint('common', 'patchsize', fallback=256)
    if image.patch_size > 0:
        forceCropSize = image.patch_size
    else:
        forceCropSize = defaultCropSize
    return render_template(template_name, project=project, image=image, defaultCropSize=defaultCropSize,forceCropSize=forceCropSize)


@html.route('/<project_name>/<image_name>/<image_id>/annotation-main.js', methods=['GET'])
def annotation_main(project_name, image_name,image_id):
    return rendered_project_image('annotation-main.js', project_name, image_name,image_id)


@html.route('/<project_name>/<image_name>/<image_id>/annotation-tool.js', methods=['GET'])
def annotation_tool(project_name, image_name,image_id):
    return rendered_project_image('annotation-tool.js', project_name, image_name,image_id)


@html.route('/<project_name>/<image_name>/<image_id>/annotation-utils.js', methods=['GET'])
def annotation_utils(project_name, image_name,image_id):
    return rendered_project_image('annotation-utils.js', project_name, image_name,image_id)


@html.route("/jobs", methods=['GET'])
@html.route("/<project_name>/jobs", methods=['GET'])
def renderprojectjob(project_name=None):
    if (project_name):
        proj = Project.query.filter_by(name=project_name).first()

        if not proj:
            return render_template("error.html")

        jobs = proj.jobs
    else:
        jobs = Job.query.all()

    return render_template('jobs.html', jobs=jobs)
