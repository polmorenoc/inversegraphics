function plywrite(fn, shape, tex, tl)
%PLYWRITE Save a head to a Stanford PLY file.
%   The PoLYgon file format (PLY) is a file format for 3D Graphics Model created by
%   Stanford University. The head is saved in binary format (litle endian). It stores the
%   3D position of each vertex (in float), the vertex color (as uchar), and the triangle
%   list.
% 
%   The PLY file format was chosen because it provides an easy way to save objects
%   represented by vertex color (as opposed to texture map which is required for the OBJ
%   file format, for instance).
%
%   More about the PLY file format at : http://local.wasp.uwa.edu.au/~pbourke/dataformats/ply
%
%Syntax:     plywrite(fn, shape, tex, tl)
%
%Inputs:
%    fn     - File name with a '.ply' extension. 
%    shape  - 3Nx1 3D shape in the interleaved format as returned by COEF2OBJECT.
%    tex    - 3Nx1 Texture as returned by COEF2OBJECT.
%    tl     - T x 3 Triangle list.
%

%% AUTHOR    : Sami Romdhani 
%% EMAIL     : Sami.Romdhani@unibas.ch 
%% URL       : http://informatik.unibas.ch/personen/romdhani_sami/ 
%% CREATION  : 18-Jul-2008 11:08:33 $ 
%% DEVELOPED : 7.6.0.324 (R2008a) 
%% FILENAME  : coef2object.m

%------------- BEGIN CODE --------------

error(nargchk(4, 4, nargin));

[fid, message] = fopen(fn, 'w');
if fid < 0, error(['Cannot open the file ' fn '\n' message]); end

nver = numel(shape)/3;
nface = numel(tl)/3;

% Writing header
fprintf(fid, 'ply\n');
fprintf(fid, 'format binary_little_endian 1.0\n');
fprintf(fid, 'comment Made from the 3D Morphable Face Model of the Univeristy of Basel, Switzerland.\n');
fprintf(fid, 'element vertex %d\n', nver);
fprintf(fid, 'property float %s\n', 'x', 'y', 'z');
fprintf(fid, 'property uchar %s\n', 'red', 'green', 'blue');
fprintf(fid, 'element face %d\n', nface);
fprintf(fid, 'property list uchar int vertex_indices\n');
fprintf(fid, 'end_header\n');

% Writing 3D shape and vertex color
for i=1:nver
  count = fwrite(fid, single(shape(3*(i-1)+1:3*i)), 'float32');
  if count ~= 3
    error('Error writing %s: %d elements were written instead of %d', fn, count, 3);
  end
  count = fwrite(fid, uint8(tex(3*(i-1)+1:3*i)), 'uchar');
  if count ~= 3
    error('Error writing %s: %d elements were written instead of %d', fn, count, 3);
  end
end

% Writing triangle list
tl = tl-1;
new_tl = zeros(3, size(tl,1), 'int32');
new_tl(1,:) = tl(:,2)';
new_tl(2,:) = tl(:,1)';
new_tl(3,:) = tl(:,3)';
nver_per_face = uint8(3);
for i=1:nface
  fwrite(fid, nver_per_face, 'uchar');
  count = fwrite(fid, new_tl(:,i), 'int32');
  if count ~= 3
    error('Error writing %s: %d elements were written instead of %d', fn, count, 3);
  end
end

fclose(fid);

%------------- END OF CODE --------------
