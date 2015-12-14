function intl = idx2intl(idx, nchannel)
%IDX2INTL convert a list of indices from the channelized format to the interleaved format
%   INTL = IDX2INTL(IDX, NCHANNEL) convert the vertex index IDX to row indexes of the MU
%   and PC matrices of the MM. NCHANNEL is the number of channel and must be 3 for the
%   shape and texture models.
%
%Example:
%   idx2intl([1; 2], 3)

% Author:      Sami Romdhani
% E-mail:      sami.romdhani@unibas.ch
% URL:         http://informatik.unibas.ch/personen/romdhani_sami/
% $Id: idx2intl.m 4586 2004-11-26 11:15:34Z romdhani $

%------------- BEGIN CODE --------------

error(nargchk(1, 2, nargin));

if nargin < 2
  nchannel = 3;
end

N = length(idx);
intl = zeros(nchannel*N, 1);
idx2 = nchannel*(idx-1);
for k=1:nchannel
  intl(k:nchannel:end) = idx2+k;
end

%------------- END OF CODE --------------
