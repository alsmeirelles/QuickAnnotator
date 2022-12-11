import os,sys
import argparse
import shutil
import QA_db
from QA_config import config, get_database_uri
from QA_db import db, Image, Project, Roi, Job

from flask import Flask

app = Flask(__name__)
app.debug = True
app.logger_name='flask'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SQLALCHEMY_DATABASE_URI'] = get_database_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = config.getboolean('sqlalchemy', 'echo', fallback=False)

APP_ROOT = os.path.dirname(os.path.abspath('__file__'))

if __name__ == '__main__': #This seems like the correct place to do this

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract annotated patches from DB and save to dir.')

    parser.add_argument('-od', dest='out_dir', type=str,default='annotated_patches', 
        help='Save selected images to this directory.')
    parser.add_argument('-n', dest='n', type=int, 
        help='Grab this many images (Default: all)', default=None,required=False)
    parser.add_argument('-p', dest='proj', type=str, default=None,required=True, 
        help='Project name.')
    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.out_dir):
        os.makedirs(config.out_dir)
        
    db.app = app
    db.init_app(app)
    db.create_all()
    db.engine.connect().execute('pragma journal_mode=wal;')

    proj_folder = f"./projects/{config.proj}/"
    proj = db.session.query(Project).filter_by(name=config.proj).first()
    if proj is None:
        print(f"No project with name: {config.proj}")
        sys.exit(-1)

    training_rois = db.session.query(Roi.id, Roi.imageId, Roi.name, Roi.path, Roi.alpath, Roi.testingROI,Roi.height, Roi.width, 
                                Roi.x, Roi.y, Roi.acq, Roi.anclass) \
            .filter(Image.projId == proj.id) \
            .filter(Roi.imageId == Image.id) \
            .filter(Roi.testingROI == 0) \
            .group_by(Roi.id).all()

    nrois = len(training_rois)
    if config.n is None or config.n >= nrois:
        config.n = nrois
        
    print("Found {} annotated ROIs.".format(nrois))
    fd = open(os.path.join(config.out_dir,'label.txt'),'w')
    
    for p in range(config.n):
        shutil.copy(training_rois[p].alpath,config.out_dir)
        fd.write("{0}\t{1}\n".format(os.path.basename(training_rois[p].alpath),training_rois[p].anclass))

    fd.close()
