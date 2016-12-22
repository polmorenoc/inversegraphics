# Overcoming Occlusion with Inverse Graphics
This is the code for our [ECCV16 paper on the Geometry meets Deep Learning workshop](http://homepages.inf.ed.ac.uk/ckiw/postscript/eccv_gmdl16.pdf). 

[img](images/vaig.png)

Here you'll find the scene generator as well as the code for the models. The main libraries needed are:

- Blender (installed as a python module), so you can do "import bpy". Please follow [these instructions](https://wiki.blender.org/index.php/User:Ideasman42/BlenderAsPyModule).
- My updated versions of OpenDR and Chumpy you can [find here](https://github.com/polmorenoc/opendr).

Python 3.4+ is a requirement.

The main script files to run are the following:

* diffrender_demo.py: Code to interactively run and fit the generative models.
* diffrender_groundtruth.py:	Main code to generate ground-truth for synthetic scenes (both for OpenGL and Photorealistic (cycles) types of rendering!
* diffrender_groundtruth_multi.p: As above, but extended to multiple objects.
* diffrender_experiment.py: Generate experiment train/test splits.
* diffrender_train.py: Train Lasagne neural networks (used to train our recognition models).
* diffrender_test.py: Main code to evaluate and fit different models.
* diffrender_analyze.py: Extract statistics and plots of experimental evaluation.

For the stocastic generation of synthetic images with ground-truth, you'll need additional CAD data and other files. Please get in touch with me (polmorenoc@gmail.com) for it.
