# SoccerKicks

The SoccerKicks dataset provides 3D guide movements for dead ball kicks (penalty and foul) obtained from reference videos suitable for use in the robotics soccer domain.

To predict the location of body joints in 3D space from monocular inputs videos, we employ the Kanazawa et al. approach [HMMR](https://github.com/nayariml/human_dynamics). We modified the HMMR system to estimate 3D poses from 2D poses provided by different 2D Human Pose Estimation models: [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

In tools you can find scripts with joint location of skeleton (2D - AlphaPose and OpenPose; 3D - HMMR and SMPL joint), draw the skeleton, to read the files, and evaluate the data.

![Schematic_overview](/overview/Diagram.png)

![gif](/overview/gif_over.gif)

### Evaluation

To evaluate the prediction we compute the average *l2 norm* on the 2D keypoints coordinates estimated for each joint given AlphaPose and OpenPose. And, we adapted the PCK (the Percentage of Correct Key-points measures the distance between the ground-truth joint location and the predicted joint location) metric from Human Pose Estimation models to compare the results of the 3D joints per frame outputs from HMMR-Alphapose and HMMR-OpenPose.

# Dataset Download

The entire contents of the dataset can be accessed through the **[link](https://drive.google.com/drive/folders/1RS93v-QE8jQ-6NFTGu4gwx_-5xYsWona?usp=sharing)**.

# Dataset content

The SoccerKicks dataset contain 38 videos with the annotations as described below:

    VideoClips 
    Rendered:
	    video clips rendered with 2D and 3D pose estimation
	    2D pose annotations for Alphapose and OpenPose system
	    3D poses annotations for Human Mesh and Motion Recovery (HMMR) system. 
    General annotations: 
    	2D_kps_info.csv
    	3D_joints_info.csv
    	dataset_evaluation.csv
    	video_source.csv          
    
    The 2D keypoints and the 3D joints location and orientation, saved in JSON files.
    Results from the HMMR system saved as pickle file:    
    
    'hmmr_output.pkl' coutain: { cams : N x 3 predicted camera, cams is 3D [s, tx, ty],
			    joints: Nx25x3 3D skeleton, refers to the 3D joint locations of the 25 keypoints,
			    kps :  N x 25 x 2 is a 2D projection, 
			    poses,   Nx24x3x3 and is a rotation matrices corresponding to the SMPL joint,
			    shapes, N x 10 shape is 10D shape coefficients of SMPL,
			    verts: N x 6890 x 3 - 3D vertices of the mesh,
			    omegas: (Bx85): [cams, poses, shapes] }
	*N is the number of frames and B referring to the number of people.

Dataset hierarchy for each video:			    
	
	id_action: 

	   annotations:
	   
	   	2D_pose_keypoints:
	   		AlphaPose_2D_kps.csv
	   		Euclidean_distance.csv
	   		OpenPose_2D_kps.csv
	   		percentage_kps.csv
	   	
	    	
	    	alphapose_hmmr_annotations: #For each joints name: 2D kps projection (x,y) and 3D joints predict (x,y,z)
	    		frame_0000_joints.json
	    		frame_0001_joints.json
	    		frame_0002_joints.json
	    		...

	    	openpose_hmmr_annotations:#For each joints name: 2D kps projection (x,y) and 3D joints predict (x,y,z)
	    		frame_0000_joints.json
	    		frame_0001_joints.json
	    		frame_0002_joints.json
	    		...

	    Alphapose_output: #2D Alphapose output

		alphapose-results-forvis-tracked.json
		alphapose-results-forvis.json
		alphapose-results.json
		2D rendered video .mp4
		vis: rendered video frames

	    hmmr_output: #3D output - 2D Alphapose_backgroud
		hmmr_output.pkl
		hmmr_output.pkl.txt

		rot_output: (joints(N X 25 X 3) and poses (N X 24 X 3 x 3))
		    joints_rot_output.json
		    joints1_rot_output.json
		    poses_rot_output.json
		    
		video_out: (3D mesh)
		    frame0000000.png
		    frame0000001.png
		    frame0000002.png
		     ...
		    hmm_output.mp4

		    hmmr_output_crop: #Renders a 2x2 video: mesh on input video, mesh on image space, 2d skel on input, and rotated mesh

		        frame0000000.png
		        frame0000001.png
		        frame0000002.png
		         ...
		        hmmr_output_crop.mp4

	    hmmr_output_openpose: #3D output - 2D Openpose_backgroud
		hmmr_output.pkl
		hmmr_output.pkl.txt

		rot_output: #joints(N X 25 X 3) and poses (N X 24 X 3 x 3)
		    joints_rot_output.json
		    joints1_rot_output.json
		    poses_rot_output.json

		video_out: (3D mesh)
		    frame0000000.png
		    frame0000001.png
		    frame0000002.png
		     ...
		    hmm_output.mp4

		    hmmr_output_crop: ()

		        frame0000000.png
		        frame0000001.png
		        frame0000002.png
		         ...

		        hmmr_output_crop.mp4

	    OpenPose_output: #2D OpenPose output

		num_action_000000000000_keypoints.json
		num_action_000000000000_rendered.jpg
		num_action_000000000001_keypoints.json
		num_action_000000000001_rendered.jpg
		 ...
		num_action.avi

	    video_frames: #videoclip frames

		frame0000000.png
		frame0000001.png
		frame0000002.png
		 ...

# License

The SoccerKicks dataset is licensed under the Creative Commons Zero v1.0 Universal, and the underlying source code used to format and display that content is licensed under the MIT license.

# Authors

Nayari Lessa, Esther Colombini and Alexandre Simões

# Citation
