%Generate a random head, render it and export it to PLY file

[model msz] = load_model();

% Generate a random head
alpha = randn(msz.n_shape_dim, 1);
beta  = randn(msz.n_tex_dim, 1);
shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );

% Render it
rp     = defrp;
rp.phi = 0.5;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = 0.6*ones(3,1);
display_face(shape, tex, model.tl, rp);


% Save it in PLY format
plywrite('rnd_head.ply', shape, tex, model.tl );


% Generate versions with changed attributes
apply_attributes(alpha, beta)

% Generate a random head with different coefficients for the 4 segments
shape = coef2object( randn(msz.n_shape_dim, msz.n_seg), model.shapeMU, model.shapePC, model.shapeEV, model.segMM, model.segMB );
tex   = coef2object( randn(msz.n_tex_dim,   msz.n_seg), model.texMU,   model.texPC,   model.texEV,   model.segMM, model.segMB );

plywrite('rnd_seg_head.ply', shape, tex, model.tl );
display_face(shape, tex, model.tl, rp);
