function apply_attributes(alpha, beta)
[model msz] = load_model();

% Render parameter
rp = defrp;
i  = 1;

% 2.d Age the face older
load ../04_attributes.mat
shape  = coef2object( alpha + 50*age_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  + 50*age_tex(1:msz.n_tex_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp); set(h, 'name', 'Older');
i=i+1; 

shape  = coef2object( alpha - 50*age_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  - 50*age_tex(1:msz.n_tex_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp);  set(h, 'name', 'Younger');
i=i+1;

% 2.e Change the gender
shape  = coef2object( alpha + 5*gender_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  + 5*gender_tex(1:msz.n_tex_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp);  set(h, 'name', 'More Male');
i=i+1; 

shape  = coef2object( alpha - 2*gender_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  - 2*gender_tex(1:msz.n_tex_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp);  set(h, 'name', 'More Female');
i=i+1;

% 2.f Change the weight
shape  = coef2object( alpha + 50*weight_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  + 50*weight_tex(1:msz.n_tex_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp);  set(h, 'name', 'Fatter');
i=i+1; 

shape  = coef2object( alpha - 30*weight_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  - 30*weight_tex(1:msz.n_tex_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp);  set(h, 'name', 'Thinner');
i=i+1;

% 2.g Older and fatter
shape  = coef2object( alpha + 50*age_shape(1:msz.n_shape_dim) + 50*weight_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta  + 50*age_tex(1:msz.n_tex_dim)     + 50*weight_tex(1:msz.n_shape_dim),     model.texMU,  model.texPC,   model.texEV );
h=figure(i); display_face(shape,tex,model.tl,rp);  set(h, 'name', 'Older and Fatter');
i=i+1;
end
