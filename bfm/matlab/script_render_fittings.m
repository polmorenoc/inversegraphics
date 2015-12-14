%Render a fitting result to show the visual quality of a fitting of the MM

%% 1. Load the model and the fitting result file
[model msz] = load_model();
load ../07_fittings/pie_lights_fres.mat

%% 2. Get the coefficients
i = find( id == 4000 & kind(1,:) == 22 & kind(2,:) == 05 );   % Find individual 4000 shot by camera 5 and lighted with flash 18
alpha = reshape(feat_mat( 100:495, i ), 99, 4);
beta  = reshape(feat_mat( 595:end, i ), 99, 4);

%% 3. Reconstruct the 3D shape add texture from the coefficients
%    Note that there are 4 sets of coefficients for the shape and the texture,
%    one for each segment of the face (nose, eyes, mouth and rest).
shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV, model.segMM, model.segMB );
tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV,   model.segMM, model.segMB );

%% 4. OPTIONAL: Load the input image used for the fitting
%    The model image will be rendered on top of this image used as background.
in_img_fn = sprintf('%02d/%05d_%02d.ppm', kind(1,i), id(i), kind(2,i));

% 5. And finally... render the image
h=figure(1);
rp     = defrp;
rp.phi = 0.5;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = 0.6*ones(3,1);
rp.sbufsize=2000;

display_face(shape,tex,model.tl,rp);
set(h, 'name', ['Fitting result: ' in_img_fn]);
