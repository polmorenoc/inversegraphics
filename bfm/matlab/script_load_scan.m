% Load and render the 10 scans provided with the MM

clear
close all

[model msz] = load_model();

% Load an example scan
fns = dir(fullfile( '..', '02_scans_matlab', '*.mat' ));
for i=1:length(fns)
  fprintf('Loading %s ...\n', fns(i).name);
  load(fullfile( '..', '02_scans_matlab', fns(i).name ));

  % Render it
  rp     = defrp;
  rp.phi = 0.5;
  rp.dir_light.dir = [0;1;1];
  rp.dir_light.intens = 0.6*ones(3,1);
  rp.sbufsize=2000;
  h=figure(1);
  display_face(shape,tex,model.tl,rp);
  set(h, 'name', 'Rendering of the scan');

  % Projection and reconstruction to/from the Morphable Model
  alpha     = object2coef( shape(:), model.shapeMU, model.shapePC, model.shapeEV );
  beta      = object2coef( tex(:),   model.texMU,   model.texPC,   model.texEV );
  shape_rec = coef2object( alpha,    model.shapeMU, model.shapePC, model.shapeEV );
  tex_rec   = coef2object( beta,     model.texMU,   model.texPC,   model.texEV );

  h=figure(2);
  display_face(shape_rec,tex_rec,model.tl,rp);
  set(h, 'name', 'Reconstruction of the 3D Scan from the Morphable Model');
  fprintf('Shape   reconstruction error : %g\n', norm(shape(:)-shape_rec(:)))
  fprintf('Texture reconstruction error : %g\n', norm(tex(:)-tex_rec(:)))

  % Projection and reconstruction to/from the Segmented Morphable Model
  alpha_seg = object2coef( shape(:),  model.shapeMU, model.shapePC, model.shapeEV, model.segbin );
  beta_seg  = object2coef( tex(:),    model.texMU,   model.texPC,   model.texEV,   model.segbin );
  shape_seg = coef2object( alpha_seg, model.shapeMU, model.shapePC, model.shapeEV, model.segMM, model.segMB );
  tex_seg   = coef2object( beta_seg,  model.texMU,   model.texPC,   model.texEV,   model.segMM, model.segMB );
  h=figure(3);
  display_face(shape_seg,tex_seg,model.tl,rp);
  set(h, 'name', 'Reconstruction of the 3D Scan from the Segmented Morphable Model');
  fprintf('Shape   reconstruction error from segmented model : %g\n', norm(shape(:)-shape_seg(:)))
  fprintf('Texture reconstruction error from segmented model : %g\n', norm(tex(:)-tex_seg(:)))

  fprintf('Hit return');
  pause
end
