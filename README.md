# SoccerKicks

The entire contents of the dataset can be accessed through the link: 

https://drive.google.com/drive/folders/1RS93v-QE8jQ-6NFTGu4gwx_-5xYsWona?usp=sharing

The dataset includes:

    Video clips 
    Rendered video clips, 2D and 3D
    2D pose annotations: Alphapose and OpenPose
    3D poses annotations: Human Mesh and Motion Recovery (HMMR) system. 
    General annotations: 
    	2D_kps_info.csv
    	3D_joints_info.csv
    	dataset_evaluation.csv
    	video_source.csv
	    
    
    Only the player is annotated
    2 types of modality : Freekick and Penalty.
    
    Dictionary results at time t saved to pickle file. Given N is the number of frames and B batch size referring to the number of people:

    'hmmr_output.pkl' coutain: { cams : N x 3 predicted camera, cams is 3D [s, tx, ty],
			    joints: Nx25x3 3D skeleton, refers to the 3D joint locations of the 25 keypoints,
			    kps :  N x 25 x 2 is a 2D projection, 
			    poses,   Nx24x3x3 and is a rotation matrices corresponding to the SMPL joint,
			    shapes, N x 10 shape is 10D shape coefficients of SMPL,
			    verts: N x 6890 x 3 - 3D vertices of the mesh,
			    omegas: (Bx85): [cams, poses, shapes] }

Dataset files:			    

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

The content of this project itself is licensed under the Creative Commons Zero v1.0 Universal, and the underlying source code used to format and display that content is licensed under the MIT license.
