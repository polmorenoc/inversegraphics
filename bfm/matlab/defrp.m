function rp = defrp
%DEFRP returns default rendering parameters.
%
%  Set in RP the default rendering parameters: front view
%  with no directed light, image size of 640 x 486.
%
%Syntax:   rp = defrp

% Author:      Sami Romdhani
% E-mail:      sami.romdhani@unibas.ch
% URL:         http://informatik.unibas.ch/personen/romdhani_sami/
% $Id: defrp.m 11524 2007-11-19 08:26:00Z romdhani $

%------------- BEGIN CODE --------------

error(nargchk(0, 0, nargin));

rp.width        = 640;
rp.height       = 486;

rp.gamma        = 0;
rp.theta        = 0;
rp.phi          = 0;
rp.alpha        = 0;
rp.t2d          = [0;0];
rp.camera_pos   = [0;0;3400];
rp.scale_scene  = 0.0;
rp.object_size  = 0.615 * 512;
rp.shift_object = [0;0;-46125];
%rp.shift_object = [0;0;0];
rp.shift_world  = [0;0;0];
rp.scale        = 0.001;

rp.ac_g         = [1; 1; 1];
rp.ac_c         = 1;
rp.ac_o         = [0; 0; 0];
rp.ambient_col  = 0.6*[1; 1; 1];

rp.rotm         = eye(3);
rp.use_rotm     = 0;

% illumination method
% 1: ac_g(1)
% 2: ac_g(1:3)
% 3: ac_g(1:3), ac_c(1), ac_o(1:3)
% 4: ambient_col(1:3), ac_g(1:3), ac_c(1), ac_o(1:3)
% 5: like 1 but it's in black and white

% do_remap = 0: no color re-mapping
% do_remap = 1: color re-mapping

rp.do_remap = 0; 
rp.dir_light = [];
%rp.dir_light.dir    = [];
%rp.dir_light.intens = [];

rp.do_specular = 0.1;
rp.phong_exp = 8;
rp.specular = 0.1*255;

rp.do_cast_shadows = 1;
rp.sbufsize = 200;

% projection method
rp.proj = 'perspective';

% if scale_scene == 0, then f is used:
rp.f = 6000;

% is 1 for grey level images and 3 for color images
rp.n_chan = 3;
rp.backface_culling = 2; % 2 = default for current projection

% can be 'phong', 'global_illum' or 'no_illum'
rp.illum_method = 'phong';

rp.global_illum.brdf = 'lambert';
rp.global_illum.envmap = struct([]);
rp.global_illum.light_probe = [];

rp.ablend = []; % no blending performed

%------------- END OF CODE --------------
