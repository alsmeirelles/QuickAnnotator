
////////////////////////////////////////////////////////////////////////////////////////////////////
function init() {
    prepareModal();
    updateImagePageButton();
    loadRunningTimers();
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function loadRunningTimers() {
    const project_id = "{{ project.id }}";
    const completed_callback_function = updateImagePageButton;
    loadRunningJobsForProject(project_id, completed_callback_function);
    pollFunc(updateALStart, 600000, 10000);
    pollFunc(updateGetPatches, 1200000, 10000);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
function pollFunc(fn, timeout, interval) {
    var startTime = (new Date()).getTime();
    interval = interval || 1000,
    canPoll = true;

    (function p() {
        canPoll = ((new Date).getTime() - startTime ) <= timeout;
        if (!fn() && canPoll)  { // ensures the function executes
            setTimeout(p, interval);
        }
    })();
}
////////////////////////////////////////////////////////////////////////////////////////////////////

function getIteration() {
    let table_name = 'project';
    let col_name = "id";
    let operation = '==';
    let value = "{{ project.id }}";

    if ( typeof getIteration.iteration == 'undefined' ) {
	getIteration.iteration = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].iteration;
	getIteration.counter = 1;
	return getIteration.iteration;
    }
    else {
	if ( getIteration.counter > 4 ) {
	    getIteration.iteration = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].iteration;
	    getIteration.counter = 0;
	}
	else {
	    getIteration.counter += 1;
	}
	return getIteration.iteration;
    
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
function updateImagePageButton() {
    //updateViewEmbed();
    //updateALStart();
    updateGenTraining();
    updateGetPatches();
    updateAnnTest();
    
    iteration = getIteration();
    if (iteration > 1) {
	document.getElementById("makePredButton").disabled = false;
	document.getElementById("makePredButton").title = "Predictions available.";
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
function displayPred() {
    addNotification("Prediction ready");
    response = JSON.parse(this.responseText);
    addNotification(`AUC: ${response.data.auc}`);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

function getProjRois() {
    let table_name = 'roi';
    let col_name = ["projId","testingROI"];
    let operation = ['==','=='];
    let value = ["{{ project.id }}",0];
    
    let rois_query = getDatabaseQueryResults(table_name, col_name, operation, value);

    return rois_query;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

function getTestingRois() {
    let table_name = 'roi';
    let col_name = ["projId","testingROI"];
    let operation = ['==','=='];
    let value = ["{{ project.id }}",1];
    
    let rois_query = getDatabaseQueryResults(table_name, col_name, operation, value);

    return rois_query;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
function updateGenTraining() {
    let table_name = 'project';
    let col_name = "id";
    let operation = '==';
    let value = "{{ project.id }}";
    let iteration = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].iteration;
    if (iteration == -1) {
        document.getElementById("initialTraining").disabled = false;
        document.getElementById("initialTraining").title = "Initial training set should be selected.";
    } else {
        document.getElementById("initialTraining").disabled = true;
        document.getElementById("initialTraining").title = "AL has already started.";
    }    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
function annotatedRois(image_id, elementID) {
    let table_name = 'roi';
    let col_name = 'imageId';
    let operation = '==';
    let rois_query = getDatabaseQueryResults(table_name, col_name, operation, image_id)
    let rois = rois_query.data.num_results;
    let rois_objects = rois_query.data.objects;

    table_name = 'project';
    col_name = "id";
    operation = '==';
    let value = "{{ project.id }}";
    let iteration = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].iteration;    

    let annotations = 0;
    for (let i = 0; i < rois; i++) {
	if (testing == 0) {
	    if((rois_objects[i].anclass >= 0) && (iteration == rois_objects[i].acq)) {
		annotations += 1;
	    }
	} else {
	    if(rois_objects[i].anclass >= 0) {
		annotations += 1
	    }
	}
    }
    document.getElementById(elementID).innerHTML = annotations;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
function updateAnnTest() {
    let rois_query = getTestingRois();
    let rois = rois_query.data.num_results;
    let rois_objects = rois_query.data.objects;

    if (rois > 0) {
	document.getElementById("annTest").disabled = false;
        document.getElementById("annTest").title = "Annotate testing patches.";
	if (testing == 0) {
	    document.getElementById("annTest").textContent = 'Annotate Test';
	    //document.querySelector('#annTest').textContent = 'Annotate Test';
	}
	else {
	    document.getElementById("annTest").textContent = 'Back to Training';
	    //document.querySelector('#annTest').textContent = 'Back to Training';
	}
    }
    else {
	document.getElementById("annTest").disabled = true;
        document.getElementById("annTest").title = "No testing patches to be annotated.";
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
function updateALStart() {
    let iteration = getIteration();

    let rois_query = getProjRois();
    let rois = rois_query.data.num_results;
    let rois_objects = rois_query.data.objects;

    if (iteration > 0 || iteration == -1) {
        document.getElementById("startAL").disabled = true;
        document.getElementById("startAL").title = "AL can not be started yet.";
    } else {
	let annotations = 0;
	addNotification(`Checking annotations for AL start.`);
	for (let i = 0; i < rois; i++) {
	    if(rois_objects[i].anclass >= 0) {
		annotations += 1;
		if(iteration == rois_objects[i].acq) {
		    elementID = `anpatches_${rois_objects[i].imageId}`;
		    obj = document.getElementById(elementID);
		    if (obj) {
			obj.innerHTML = 1;
 		    }
		}
	    }
	}
	if (annotations == rois) {
            document.getElementById("startAL").disabled = false;
            document.getElementById("startAL").title = "AL can be started.";
	    return true;
	}
	else{
	    addNotification(`Not all patches have been annotated (${rois-annotations}).`);
	    return false;
	}
    }    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
function updateGetPatches() {
    let iteration = getIteration();

    let rois_query = getProjRois();
    let rois = rois_query.data.num_results;
    let rois_objects = rois_query.data.objects;

    if (iteration < 1) {
        document.getElementById("getALBatch").disabled = true;
        document.getElementById("getALBatch").title = "Finish annotating current patches first.";
    } else {
	let annotations = 0;
	addNotification(`Checking annotations before new patches can be acquired.`);
	for (let i = 0; i < rois; i++) {
	    if(rois_objects[i].anclass >= 0) {
		annotations += 1;
		if(iteration == rois_objects[i].acq) {
		    elementID = `anpatches_${rois_objects[i].imageId}`;
		    obj = document.getElementById(elementID);
		    if (obj) {
			obj.innerHTML = 1;
		    }
 		}
	    }
	}
	if (annotations == rois) {
            document.getElementById("getALBatch").disabled = false;
            document.getElementById("getALBatch").title = "A new patch set can be acquired.";
	    return true;
	}
	else{
	    addNotification(`Not all patches have been annotated (${rois-annotations}).`);
	    return false;
	}
    }        

}
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
function updateMakePatches() {
    let table_name = 'image';
    let col_name = 'projId';
    let operation = '==';
    let value = "{{ project.id }}";
    let numImage = getDatabaseQueryResults(table_name, col_name, operation, value).data.num_results;
    if (numImage == 0) {
        document.getElementById("makePacthButton").disabled = true;
        document.getElementById("makePacthButton").title = "'Make Patches' is NOT ready to use."
    } else {
        document.getElementById("makePacthButton").disabled = false;
        document.getElementById("makePacthButton").title = "'Make Patches' is ready to use."
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function updateTrainAE() {
    let table_name = 'project';
    let col_name = 'id';
    let operation = '==';
    let value = "{{ project.id }}";
    let make_patches_time = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].make_patches_time;
    if (make_patches_time == null) {
        document.getElementById("trainAEButton").disabled = true;
        document.getElementById("trainAEButton").title = "'(Re)train Model 0' is NOT ready to use."
    } else {
        document.getElementById("trainAEButton").disabled = false;
        document.getElementById("trainAEButton").title = "'(Re)train Model 0' is ready to use."
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function updateMakeEmbed() {
    let table_name = 'project';
    let col_name = 'id';
    let operation = '==';
    let value = "{{ project.id }}";
    let train_ae_time = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].train_ae_time;
    let iteration = getIteration();
    if (iteration == -1) {
        document.getElementById("makeEmbedButton").disabled = true;
        document.getElementById("makeEmbedButton").title = "'Embed Patches' is NOT ready to use.";
    } else if (train_ae_time == null && iteration ==0){
        document.getElementById("makeEmbedButton").disabled = true;
        document.getElementById("makeEmbedButton").title = "The latest model is model 0. The Model 0 is being retrained. No other DL model is available at this moment. \n" +
            "Make_embed is currently unavailable"
    }
    else {
        document.getElementById("makeEmbedButton").disabled = false;
        document.getElementById("makeEmbedButton").title = "'Embed Patches' is ready to use.";
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function updateViewEmbed() {
    let table_name = 'project';
    let col_name = 'id';
    let operation = '==';
    let value = "{{ project.id }}";
    let embed_iteration = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].embed_iteration;
    if (embed_iteration == -1) {
        document.getElementById("viewEmbedButton").disabled = true;
        document.getElementById("viewEmbedButton").title = "'View Embedding' is NOT ready to use.";
    } else {
        document.getElementById("viewEmbedButton").disabled = false;
        document.getElementById("viewEmbedButton").title = "'View Embedding' is ready to use.";
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function start_al() {
    addNotification("'Start AL' pressed. Active Learning system will start.")
    addNotification("'Start AL' pressed. This can take up to 15 minutes.")
 
    const run_url = new URL("{{ url_for('api.get_al_patches', project_name=project.name) }}", window.location.origin);
    return loadObjectAndRetry(run_url, reload_images)
}
////////////////////////////////////////////////////////////////////////////////////////////////////

function make_prediction() {
    addNotification("'Make Prediction' pressed. Test set will be evaluated.")
    addNotification("Predictions can take some minutes.")

    let xhr = new XMLHttpRequest();
    let run_url = "{{ url_for('api.make_al_prediction', project_name=project.name) }}";
    xhr.onreadystatechange = function () {
	if (this.readyState == 4 && this.status == 200) {
	    addNotification("Prediction ready");
	    response = JSON.parse(this.responseText);
	    document.getElementById("pred").innerHTML = `AUC: ${response.data.auc};   Accuracy: ${response.data.acc}`;
	}
    }

    xhr.open("GET", run_url, true);
    xhr.send();
    
    //return loadObjectAndRetry(run_url, displayPred)
}
////////////////////////////////////////////////////////////////////////////////////////////////////

function get_al_batch() {
    addNotification("'Get Patch Set' Pressed. Active Learning system will acquire a new patch set for annotation.")
    addNotification("This may take a few minutes...")

    document.getElementById("makePredButton").disabled = true;
    document.getElementById("makePredButton").title = "Wait for new batch.";
    
    pollFunc(updateGetPatches, 300000, 10000);
    const run_url = new URL("{{ url_for('api.get_al_patches', project_name=project.name) }}", window.location.origin);
    return loadObjectAndRetry(run_url, reload_images)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

function annotateTest() {

    if (testing == 0) {
	addNotification("Showing test set patches for annotation.")

	document.getElementById("getALBatch").disabled = true;
	document.getElementById("getALBatch").title = "You are annotating the test set.";
	document.getElementById("startAL").disabled = true;
	document.getElementById("startAL").title = "You are annotating the test set.";
	
	const run_url = new URL("{{ url_for('html.get_imagelist', project_name=project.name, testing=1) }}", window.location.origin);
	window.location.href = run_url;
	//return loadObjectAndRetry(run_url)	
    } else {
	pollFunc(updateGetPatches, 300000, 10000);
	document.getElementById("annTest").innerText = 'Annotate Test';
	
	const run_url = new URL("{{ url_for('html.get_imagelist', project_name=project.name) }}", window.location.origin);
	window.location.href = run_url;
	//return loadObjectAndRetry(run_url)
    }
    return true
}

////////////////////////////////////////////////////////////////////////////////////////////////////
function train_ae() {
    addNotification("'(Re)train Model 0' Pressed.")
    if (checkMake_embed()) {
        addNotification("The latest model is model 0. The Model 0 is being retrained. No other DL model is available at this moment. \n" +
            "Make_embed is currently unavailable")
        document.getElementById("makeEmbedButton").disabled = true;
        document.getElementById("makeEmbedButton").title = "The latest model is model 0. The Model 0 is being retrained. No other DL model is available at this moment. \n" +
            "Make_embed is currently unavailable"
    }
    const run_url = new URL("{{ url_for('api.train_autoencoder', project_name=project.name) }}", window.location.origin);
    return loadObjectAndRetry(run_url, updateImagePageButton)
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function reload_images() {
    addNotification("Reloading selected images.")
    location.reload()
    updateImagePageButton()
}

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function generate_train() {
    addNotification("'Generate Training Set' Pressed.This could take some minutes.")

    const run_url = new URL("{{ url_for('api.generate_train', project_name=project.name) }}", window.location.origin);
    return loadObjectAndRetry(run_url, reload_images)
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function make_patches() {
    addNotification("'Make Patches' Pressed.")
    // Using URL instead of string here
    const run_url = new URL("{{ url_for('api.make_patches', project_name=project.name) }}", window.location.origin)
    let $dialog = $('<div></div>').html('SplitText').dialog({
        dialogClass: "no-close",
        modal: true,
        title: "Make Patches",
        // We have three options here
        buttons: {
            // Remove the white background when making patches
            "Remove": function () {
                run_url.searchParams.append("whiteBG", "remove")
                $dialog.dialog('close');
                addNotification("'Make Patches' (White Background Removed) starts.")
                return loadObjectAndRetry(run_url, updateImagePageButton)
            },
            // Keep the white backgeound when making patches
            "Keep": function () {
                $dialog.dialog('close');
                addNotification("'Make Patches' (White Background Kept) starts.")
                return loadObjectAndRetry(run_url, updateImagePageButton)
            },
            // Simply close the dialog and return to original page
            "Cancel": function () {
                addNotification("'Make Patches' cancels.")
                $dialog.dialog('close');
            }
        }
    });
    $dialog.html("Do you want to remove the white background from the patches?")
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function make_embed() {
    addNotification("'Embed Patches' Pressed.")
    const run_url = new URL("{{ url_for('api.make_embed', project_name=project.name) }}", window.location.origin)
    return loadObjectAndRetry(run_url, updateImagePageButton);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
function checkMake_embed() {
    let table_name = 'project';
    let col_name = 'id';
    let operation = '==';
    let value = "{{ project.id }}";
    let train_ae_time = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].train_ae_time;
    let iteration = getDatabaseQueryResults(table_name, col_name, operation, value).data.objects[0].iteration;
    // Latest model is model 0 and model 0 is being retrain.
    if (train_ae_time != null && iteration == 0) {
        return true;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
function delete_image(image_name) {
    let xhr = new XMLHttpRequest();
    let $dialog = $('<div></div>').html('SplitText').dialog({
        dialogClass: "no-close",
        modal: true,
        title: "Delete Image",
        buttons: {
            "Delete": function () {
                $dialog.dialog('close');
                addNotification(`Delete the image: '${image_name}'.`);
                let run_url = "{{ url_for('api.delete_image', project_name = project.name, image_name= '') }}" + image_name;
                xhr.onreadystatechange = function () {
                    if (this.readyState == 1 && this.status == 0) {
                        // This is to display the block of the image as none; it is kept multiple browsers
                        document.getElementById(image_name).style.display = "none";
                        // Remove the html linked to the deleted image
                        document.getElementById(image_name).outerHTML = "";
                    }
                };
                xhr.open("DELETE", run_url, true);
                xhr.send();

            },
            // Simply close the dialog and return to original page
            "Cancel": function () {
                addNotification(`Delete Image '${image_name}' cancels.`)
                $dialog.dialog('close');
            }
        }
    });
    $dialog.html("Do you want to delete the selected image?")
}
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
function updatePercentCompleted(width, height, ppixel, npixel, elementID) {
    let totalPixel = Number(width) * Number(height) / 100 // For stability concern
    let annotatedPixel = Number(ppixel) + Number(npixel)
    let percent_completed = annotatedPixel / totalPixel;
    document.getElementById(elementID).innerHTML = Math.ceil(percent_completed) + "%";
}
////////////////////////////////////////////////////////////////////////////////////////////////////
