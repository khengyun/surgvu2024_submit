# Docker image & algorithm submission for Category 1 of SurgToolLoc Challenge 2024


- Weight: https://github.com/khengyun/surgvu2024_submit/blob/main/src/models/best.pt
- Dataset: https://drive.google.com/file/d/1ZarrwoFOhYUudDZVV2OOTZcyU5F2DPVO

## Prerequisites

You will need to have [Docker](https://docs.docker.com/) installed on your system. We recommend using Linux with a Docker installation. If you are on Windows, please use [WSL 2.0](https://docs.microsoft.com/en-us/windows/wsl/install).


## Prediction format

For category 1 of [SurgToolLoc Challenge](https://surgtoolloc.grand-challenge.org/) (surgical tool detection) the instructions to generate the Docker container are given below

### Category #1 – Surgical tool detection:  

The output json file needs to be a dictionary containing the set of tools detected in each frame with its correspondent bounding box corners (x, y), again generating a single json file for each video like given below:  

```
{ 
    "type": "Multiple 2D bounding boxes", 
    "boxes": [ 
        { 
        "corners": [ 
            [ 54.7, 95.5, 0.5], 
            [ 92.6, 95.5, 0.5], 
            [ 92.6, 136.1, 0.5], 
            [ 54.7, 136.1, 0.5] 
        ], 
        "name": "slice_nr_1_needle_driver",
        "probability": 0.452
        }, 
        { 
        "corners": [ 
            [ 54.7, 95.5, 0.5], 
            [ 92.6, 95.5, 0.5], 
            [ 92.6, 136.1, 0.5], 
            [ 54.7, 136.1, 0.5] 
        ], 
        "name": "slice_nr_2_monopolar_curved_scissor", 
        "probability": 0.783
        } 
    ], 
    "version": { "major": 1, "minor": 0 } 
} 
```
 Please note that the third value of each corner coordinate is not necessary for predictions but must be kept 0.5 always to comply with the Grand Challenge automated evaluation system (which was built to also consider datasets of 3D images). To standardize the submissions, the first corner is intended to be the top left corner of the bounding box, with the subsequent corners following the clockwise direction. The “type” and “version” entries are to comply with grand-challenge automated evaluation system. 
 **Please use the "probability" entry to include the confidence score for each detected bounding box.**


## Adapting the container to your algorithm

1. First, clone this repository:

```
git clone https://github.com/khengyun/surgvu2024_submit.git
```

2. Our `Dockerfile` should have everything you need, but you may change it to another base image/add your algorithm requirements if your algorithm requires it.


3. Add your trained model weights (e.g., `best.pt`) to the `src/models/` directory.

4. The `process.py` script is the main step for adapting this repository for your model. This script will load your model and corresponding weights, perform inference on input videos one by one along with any required pre/post-processing, and return the predictions of surgical tool classification as a dictionary. The class `Surgtoolloc_det` contains the `predict` function. You should replace the dummy code in this function with the code for your inference algorithm. Use `__init__` to load your weights and/or perform any needed operations. We have added `TODO` on the places you need to adapt for your model.

5. You can use the `taskfile` to manage different parts of the workflow easily. Below are the instructions for common operations:

   - **Initialize the environment**: Run the following to install dependencies, including `ultralytics` and `yolov10`:
     ```
     task init
     ```

   - **Move the model to the submission directory**: Use this command to move your model from the training output directory to the `src/models/` folder:
     ```
     task model_move
     ```

   - **Build the Docker container**: To build the Docker container, run:
     ```
     task build
     ```

   - **Run local testing**: If you wish to run local tests, ensure your `test.sh` script is correctly configured and then execute:
     ```
     task test
     ```

   - **Run the Python script locally**: If you want to test the `process.py` script directly, use:
     ```
     task local
     ```

   - **Export the container for submission**: To create the `.tar.gz` file needed for submission to the Grand Challenge platform, execute:
     ```
     task submit
     ```

6. You can modify the `test.sh` script and parts of `process.py` to adapt for your local testing. The main check is whether the output JSON produced by your algorithm container in `./output/surgical-tools.json` is similar to the sample JSON in the repository (also named `surgical-tools.json`).

7. Once you're satisfied with the results, run the export script to package your container for submission:
    ```
    task submit
    ```


8. Follow the steps outlined in the "Uploading your container to the grand-challenge platform" section to submit your container to the SurgToolLoc Challenge.



## Uploading your container to the grand-challenge platform

1. Create a new algorithm [here](https://surgtoolloc.grand-challenge.org/evaluation/challenge/algorithms/create/). Fill in the fields as specified on the form.

2. On the page of your new algorithm, go to `Containers` on the left menu and click `Upload a Container`. Now upload your `.tar.gz` file produced in step 7. 

3. After the Docker container is marked as `Ready`, you may be temped to try out your own algorithm when clicking `Try-out Algorithm` on the page of your algorithm. But doing so will likely fail. WARNING: Using this container in `Try-out` will fail. You can still use the Try-out feature to check logs from the algorithm and ensure that processes are running but it will not pass. However, if built correctly and you see the expected logs from your algorithm, then the container should still work for the Prelim submission. 

4. WE STRONGLY RECOMMEND that you make at least 1-2 Prelim submissions before August 26th to ensure that your container runs correctly. Start earlier (Aug 19th) so we can help debug issues that may arise, otherwise there will be no opportunities to debug containers during the main submission!

5. To make a submission to one of the test phases. Go to the [SurgToolLoc Challenge](https://surgtoolloc.grand-challenge.org/) and click `Submit`. Under `Algorithm`, choose the algorithm that you just created. Then hit `Save`. After the processing in the backend is done, your submission should show up on the leaderboard if there are no errors.

The figure below indicates the step-by-step of how to upload a container:

![Alt text](README_files/MICCAI_surgtoolloc_fig.png?raw=true "Flow")

If something does not work for you, please do not hesitate to [contact us](mailto:isi.challenges@intusurg.com) or [add a post in the forum](https://grand-challenge.org/forums/forum/endoscopic-surgical-tool-localization-using-tool-presence-labels-663/). 

## Acknowledgments

The repository is greatly inspired and adapted from [MIDOG reference algorithm](https://github.com/DeepPathology/MIDOG_reference_docker), [AIROGS reference algorithm](https://github.com/qurAI-amsterdam/airogs-example-algorithm) and [SLCN reference algorithm](https://github.com/metrics-lab/SLCN_challenge)

