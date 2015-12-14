function obj = coef2object(coef, mu, pc, ev, MM, MB)
%COEF2OBJECT produces a shape or a texture from MM coefficient.
%   OBJ = COEF2OBJECT(COEF, MU, PC, EV) produce a shape or a texture OBJ from coefficients
%   COEF given a model. The model can be a shape or texture model. It is given as three
%   matrices. The vector MU stores the mean, the matrix PC, the principal components and
%   EV the standard deviation of each principal components. The coefficients COEF must be
%   provided in units of standard deviation, i.e.  the range of values is the same for all
%   coefficients and is comparable to 1. 
%
%   OBJ = COEF2OBJECT(COEF, MU, PC, EV, MM, MB) produce a shape or texture from
%   coefficients of 4 segments. The different segments are blended together to yield a
%   single shape or texture. A segment is a portion of a face. In the MM four segments are
%   used: the nose, the eyes, the mouth and the rest. The coefficient of a segment is
%   given as a colum vector of COEF. Hence, COEF is a matrix with four columns. 
%
%   The number or rows of COEF must be equal to the number of columns of PC and to the
%   number of elements of EV. The number of rows of MU must be equal to the number of rows
%   of PC.
%
%Example:
%   alpha = randn(99, 4);   % Random shape   coefficients of 4 segments
%   beta  = randn(99, 4);   % Random texture coefficients of 4 segments
%   shape  = coef2object( alpha, shapeMU, shapePC, shapeEV, MM, MB );
%   tex    = coef2object( beta,  texMU,   texPC,   texEV,   MM, MB );
%
%See also OBJECT2COEF SCRIPT_GEN_RANDOM SCRIPT_RENDER_FITTING

%% AUTHOR    : Sami Romdhani 
%% EMAIL     : Sami.Romdhani@unibas.ch 
%% URL       : http://informatik.unibas.ch/personen/romdhani_sami/ 
%% CREATION  : 18-Jul-2008 11:08:33 $ 
%% DEVELOPED : 7.6.0.324 (R2008a) 
%% FILENAME  : coef2object.m

%------------- BEGIN CODE --------------

% Arguments checking
if nargin ~= 4 && nargin ~= 6
  error('Inappropriate number of arguments')
end

if nargout ~= 1
  error('One output argument required')
end

n_seg = size(coef, 2);
if nargin == 4 && n_seg > 1
  error('Blending reconstruction requested, but blending parameters missing')
end

n_dim = size(coef, 1);
if n_dim > size(pc, 2)
  error('Too many coefficients.')
end

% Reconstruction
obj = mu*ones([1 n_seg]) + pc(:,1:n_dim) * (coef .* (ev(1:n_dim)*ones([1 n_seg])) );
if nargin == 4, return; end

% Blending (optional)
n_ver = size(obj,1)/3;
all_vertices = zeros(n_seg*n_ver, 3);
k=0;
for i=1:n_seg
  all_vertices(k+1:k+n_ver, :) = reshape(obj(:,i), 3, n_ver)';
  k = k+n_ver;
end
clear obj k
obj = (MM \ (MB*all_vertices) )';
obj = obj(:);

%------------- END OF CODE --------------
