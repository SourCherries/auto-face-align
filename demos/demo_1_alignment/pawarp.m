function [warpim tri inpix fwdwarpix] = pawarp(im,base,target,varargin)
% function [warpim tri inpix fwdwarpix] = pawarp(im,base,target,varargin)
% 
%  piecewise affine warp of target image ''im'' from target coords to base coords
%
%  Required inputs
%    im - [ypix X xpix (X 3)] image to be warped
%    base - [nverts X 2] pixel coordinates of base mesh (left->right, bottom->top)
%    target - [nverts X 2] pixel coordinates of target mesh (left->right,
%    bottom->top)
%
%  Optional inputs (string,value pairs)
%    ''tri'' - [nTriangles X 3] precomputed Delaunay triangulation of
%        base mesh (default performs Delaunay triangulation on base mesh).
%    ''inpix'' - {[nPix X 2]; [nPix X 1]} cell array
%        containing: precomuted array of pixel coordinates; precomputed triangle location of each
%        pixel in array (default is all pixels inside square bounds of base
%        mesh, and uses ''pointLocation'' to find triangle membership per
%        pixel).  Required if manually providing precomputed Delauany
%        triangles.
%    ''interp'' - ''bilin'' = bilinear interpolation or ''nearest'' = rounds to nearest pixel 
%        location (default = ''bilin'').
%    ''doim'' - ''yes'' (default) or ''no''.  Tell function whether to
%        actually compute pixel values, or just return forward warp
%        locations.
%    ''fullim'' - ''yes'' or ''no'' (defualt).  Tell function whether to
%        return warp pixels over the full image or just that part inside the convex hull
%    ''imdims'' - [y x] (default = ypix xpix]).
%    ''subdivide'' - integer number of subdivision loops to perform prior
%        to warping (default = 0)
%
%  Outputs
%    warpim - [ypix*ratio(2) X xpix*ratio(1)] warped image
%    tri - [nTriangles X 3] Delaunay triangulation of base mesh
%    inpix - {[nPix X 2]; [nPix X 1]} cell array
%        containing: array of pixel coordinates; triangle location of each
%        pixel in array
%    fwdwarpix - [nPix X 2] array of forward warp locations for all pixels
%        in ''inpix''
%
% Oliver G B Garrod 08/10/09
% version 1.03
%

if (nargin < 3)
    error('pawarp: at least three inputs required.  See ''help pawarp''.');
end


%set optional arguments and defaults
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optargin = length(varargin);

if rem(optargin,2)
    error('pawarp: incorrect optional input format.  Should be a set of string/value pairings');
end
optargin = optargin/2;

s = cell(optargin,1);
v = cell(optargin,1);
for i = 1:optargin
    s{i} = lower(varargin{2*i-1});
    v{i} = varargin{2*i};
end

gotarg = zeros(optargin,1);

isdoim = strmatch('doim',s,'exact');
if (~isempty(isdoim))
    gotarg(isdoim) = 1;
    doim = v{isdoim};
    ndims = length(size(doim));
    [a b] = size(doim);
    if (ndims > 2 || (a~=1 && b~=1) || ~ischar(doim))
       error('pawarp: ''doim'' should be a string valued input.'); 
    end
    yes = strmatch('yes',lower(doim),'exact');
    no = strmatch('no',lower(doim),'exact');
    if (~isempty(yes))
        doim = 1;
    elseif (~isempty(no))
        doim = 0;
    else
        error('pawarp: ''doim'' should be either of ''yes'' or ''no''.');
    end
else
    doim = 1;
    isdoim = 0;
end

isfullim = strmatch('fullim',s,'exact');
if (~isempty(isfullim))
    gotarg(isfullim) = 1;
    fullim = v{isfullim};
    ndims = length(size(fullim));
    [a b] = size(fullim);
    if (ndims > 2 || (a~=1 && b~=1) || ~ischar(fullim))
       error('pawarp: ''fullim'' should be a string valued input.'); 
    end
    yes = strmatch('yes',lower(fullim),'exact');
    no = strmatch('no',lower(fullim),'exact');
    if (~isempty(yes))
        fullim = 1;
    elseif (~isempty(no))
        fullim = 0;
    else
        error('pawarp: ''fullim'' should be either of ''yes'' or ''no''.');
    end
else
    fullim = 0;
    isfullim = 0;
end


isinterp = strmatch('interp',s,'exact');
if (~isempty(isinterp))
    gotarg(isinterp) = 1;
    interp = v{isinterp};
    ndims = length(size(interp));
    [a b] = size(interp);
    if (ndims > 2 || (a~=1 && b~=1) || ~ischar(interp))
       error('pawarp: ''interp'' should be a string valued input.'); 
    end
    bil = strmatch('bilin',lower(interp),'exact');
    off = strmatch('nearest',lower(interp),'exact');
    if (~isempty(bil))
        bl = 1;
    elseif (~isempty(off))
        bl = 0;
    else
        error('pawarp: ''interp'' should be either of ''bilin'' or ''nearest''.');
    end
else
    bl = 1;
    isinterp = 0;
end


istri = strmatch('tri',s,'exact');
if (~isempty(istri))
    gotarg(istri) = 1;
    tri = v{istri};
    ntri = size(tri,1);
    ndims = length(size(tri));
    [a b] = size(tri);
    if (ndims > 2 || b ~= 3)
        error('pawarp: triangulation data should be of the form [nTriangles X 3]');
    end
else
    istri = 0;
end

isimdims = strmatch('imdims',s,'exact');
if (~isempty(isimdims))
    gotarg(isimdims) = 1;
    if (~isvector(v{isimdims}))
        error('pawarp: ''imdims'' should be a [2 X 1] vector.');
    end
    ndims = length(size(v{isimdims}));
    [a b] = size(v{isimdims});
    if (ndims > 2 || (a ~= 1 && b~=1))
        error('pawarp: ''imdims'' should be a [2 X 1] vector.');
    end
    
    imdims = v{isimdims};
    
else
    isimdims = 0;
end

isinpix = strmatch('inpix',s,'exact');
if (~isempty(isinpix))
    gotarg(isinpix) = 1;
    if (~iscell(v{isinpix}))
        error('pawarp: ''inpix'' should be a [2 X 1] cell array.');
    end
    ndims = length(size(v{isinpix}));
    [a b] = size(v{isinpix});
    if (ndims > 2 || (a ~= 1 && b~=1))
        error('pawarp: ''inpix'' should be a [2 X 1] cell array.');
    end
    
    ndims = length(size(v{isinpix}{1}));
    [a b] = size(v{isinpix}{1});
    if (ndims > 2 || b~=2)
        error('pawarp: first entry of ''inpix'' should be a [nPix X 2] array of pixel coordinates.');
    end

    ndims = length(size(v{isinpix}{2}));
    [a b] = size(v{isinpix}{2});
    if (ndims > 2 || (a~=1 && b~=1) || (a~=size(v{isinpix}{1},1) && b~=size(v{isinpix}{1},1)))
        error('pawarp: second entry of ''inpix'' should be a [nPix X 1] vector of triangle membership per pixel in pixel array.');
    end
    
    pixarr = v{isinpix}{1};
    inpix = v{isinpix}{2};    
else
    isinpix = 0;
end

issubdivide = strmatch('subdivide',s,'exact');
if (~isempty(issubdivide))
    gotarg(issubdivide) = 1;
    ndims = length(size(v{issubdivide}));
    [a b] = size(v{issubdivide});
    if ndims > 2 || a~=1 || b~=1 || mod(v{issubdivide},1)~=0
        error('pawarp: ''subdivide'' should be a scalar integer.');
    end
    
    nsub = v{issubdivide};
    
else
    nsub = 0;
end


if (sum(gotarg) < optargin)
    argstr = '';
    for arg = 1:sum(~gotarg)
       argstr = [argstr ' ''%s''']; 
    end
    warnstr = ['pawarp: optional arguments' argstr ' were not recognized so they were ignored.'];
    warning(warnstr, s{~gotarg});
end

if istri && ~isinpix, error('pawarp: if triangle faces are provided, pixel locations must be provided also\n'); end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




isuint8 = false;
if strcmp(class(im),'uint8') || strcmp(class(im),'uint16')
    isuint8 = true;
end
% check that image dimensions are legal and deal with greyscale vs RGB
if (length(size(im)) > 3)
    error('pawarp: image should be either [xpix X ypix] or [xpix X ypix X 3]');
elseif (length(size(im)) == 3) % allocate array for warped output (same size as target image)
    nRGB = size(im,3);
    im = single(im);
else
    nRGB = 1;
end

if ~isimdims
    imdims = [size(im,1) size(im,2)];
else
    if max(target(:)) > 1
        target = [target(:,1)/size(im,2) 1-(target(:,2)/size(im,1))];
        target = [target(:,1)*imdims(2) (1-target(:,2))*imdims(1)];
    end
    im = imresize(im, imdims, 'method', 'bilinear');
end

% figure, imshow(im)
% hold on
% scatter(target(:,1),target(:,2), 'filled');

warpim = zeros([imdims nRGB],'single');

nverts = size(base,1);
nverts2 = size(target,1);

if (nverts2 ~= nverts)
    error('pawarp: number of vertices in base and target must match');
end

if max(base(:)) <= 1
    if max(target(:)) > 1
        error('pawarp: both base and target coords should be in the same format');
    end
    base = [base(:,1)*imdims(2) (1-base(:,2))*imdims(1)];
    target = [target(:,1)*imdims(2) (1-target(:,2))*imdims(1)];
end

if fullim
    boxpix = [1 1; 1 imdims(1); imdims(2) 1; imdims(2) imdims(1)];
else
    boxpix = [];
end
% base vertices in pixel coordinates
pix1 = double([base]);%.* repmat(imdims([2 1]), [nverts 1]);
% target vertices in pixel coordinates
pix2 = double([target]);%.* repmat(imdims([2 1].*ratio), [nverts 1]);

if nsub > 0
    % subdivide the mesh
    if ~istri
        dt = DelaunayTri(pix1(:,1),pix1(:,2));
        tri = dt.Triangulation;
    end
    if isempty(which('subdivisionloop'))
        warning('pawarp: ''subdivisionloop'' function is missing.  Skipping subdivision');
    else
        for i = 1:nsub
            [p,t] = subdivisionloop([pix1 zeros(size(pix1,1),1)],tri);
            pix1 = p(:,1:2);
            [p,~] = subdivisionloop([pix2 zeros(size(pix2,1),1)],tri);
            pix2 = p(:,1:2);
            tri = t;
        end
        pix1 = [pix1;boxpix];
        pix2 = [pix2;boxpix];
        dt = DelaunayTri(pix1(:,1),pix1(:,2));
        tri = dt.Triangulation;
    end
    ntri = size(tri,1);
else
    pix1 = [pix1;boxpix];
    pix2 = [pix2;boxpix];
    if (~istri)
        % perform Delaunay triangulation on pixel coordinates of base vertices
        dt = DelaunayTri(pix1(:,1),pix1(:,2));
        tri = dt.Triangulation;
        ntri = size(tri,1);
    end
end



% get the first, second, and third vertex for each triangle (x--coords)
xio = pix1(tri(:,1),1);
xi = pix2(tri(:,1),1);
xjo = pix1(tri(:,2),1);
xj = pix2(tri(:,2),1);
xko = pix1(tri(:,3),1);
xk = pix2(tri(:,3),1);
% get the first, second, and third vertex for each triangle (y--coords)
yio = pix1(tri(:,1),2);
yi = pix2(tri(:,1),2);
yjo = pix1(tri(:,2),2);
yj = pix2(tri(:,2),2);
yko = pix1(tri(:,3),2);
yk = pix2(tri(:,3),2);

% array for warp parameters (one set of params per triangle)
wparams = zeros(ntri,6);

% calculate warp parameters for each triangle
denom = (xjo-xio).*(yko-yio) - (yjo-yio).*(xko-xio);
wparams(:,1) = ( xio.*( (xk-xi).*(yjo-yio) - (xj-xi).*(yko-yio) ) ...
    + yio.*( (xj-xi).*(xko-xio) - (xk-xi).*(xjo-xio) ) ) ...
    ./ denom ...
    + xi;
wparams(:,4) = ( xio.*( (yk-yi).*(yjo-yio) - (yj-yi).*(yko-yio) ) ...
    + yio.*( (yj-yi).*(xko-xio) - (yk-yi).*(xjo-xio) ) ) ...
    ./ denom ...
    + yi;
wparams(:,2) = ( (xj-xi).*(yko-yio) - (xk-xi).*(yjo-yio) ) ...
    ./ denom;
wparams(:,5) = ( (yj-yi).*(yko-yio) - (yk-yi).*(yjo-yio) ) ...
    ./ denom;
wparams(:,3) = ( (xk-xi).*(xjo-xio) - (xj-xi).*(xko-xio) ) ...
    ./ denom;
wparams(:,6) = ( (yk-yi).*(xjo-xio) - (yj-yi).*(xko-xio) ) ...
    ./ denom;



if (~isinpix)
    % determine square bounds of pixels inside base mesh
    xmx = min([ceil(max(pix1(:,1))) imdims(2)]);
    xmn = max([floor(min(pix1(:,1))) 1]);
    ymx = min([ceil(max(pix1(:,2))) imdims(1)]);
    ymn = max([floor(min(pix1(:,2))) 1]);

    pixarr = zeros(numel(im(ymn:ymx,xmn:xmx,1)),2); % array for pixel coordinates inside base mesh
    cnt = 0;
    % generate all possible pixel coordinates inside square bounds of base mesh
    for i = ymn:ymx
        for j = xmn:xmx
            cnt = cnt+1;
            pixarr(cnt,1) = j;
            pixarr(cnt,2) = i;
        end
    end

    % determine triangle that each possible pixel falls under
    inpix = pointLocation(dt,pixarr);
end

% get only those pixels that are inside the convex hull
isin = find(~isnan(inpix));



wp = wparams(inpix(isin),:); % warp parameters for each pixel inside convex hull


% calculate pixel coordinates in target image of each base pixel inside convex hull
if (bl)
    fwdwarpix = [
        wp(:,1) + wp(:,2).*pixarr(isin,1) + wp(:,3).*pixarr(isin,2) ...
        wp(:,4) + wp(:,5).*pixarr(isin,1) + wp(:,6).*pixarr(isin,2)
        ];
else
    fwdwarpix = round([
        wp(:,1) + wp(:,2).*pixarr(isin,1) + wp(:,3).*pixarr(isin,2) ...
        wp(:,4) + wp(:,5).*pixarr(isin,1) + wp(:,6).*pixarr(isin,2)
        ]);
end

if doim
    fwdwarpix(fwdwarpix<1) = 1;
    fwdwarpix(isnan(fwdwarpix)) = 1;
    fwdwarpix(fwdwarpix(:,1)>imdims(2),1) = imdims(2);
    fwdwarpix(fwdwarpix(:,2)>imdims(1),2) = imdims(1);
    RGBsub = [];
    for RGB = 1:nRGB
        RGBsub = cat(1, RGBsub,ones(size(fwdwarpix,1),1)*RGB);
    end
    if ~bl
        fwdwarpind = sub2ind([imdims nRGB], repmat(fwdwarpix(:,2),[nRGB 1]),repmat(fwdwarpix(:,1),[nRGB 1]), RGBsub);
    end
    pixind = sub2ind([imdims nRGB], repmat(pixarr(isin,2),[nRGB 1]),repmat(pixarr(isin,1),[nRGB 1]), RGBsub);
end


if (doim)
    % set the image intensity of each base pixel inside convex hull to that of
    % its corresponding target pixel
    if (bl)
        warpim(pixind) = bilin(im,fwdwarpix, nRGB,RGBsub);
    else
        warpim(pixind) = im(fwdwarpind);
    end
end

inpix = {pixarr; inpix};
if (isuint8)
    warpim = uint8(warpim);
end


%%%%%%%%%%%%%% bilinear interpolation %%%%%%%%%%%%%%%%%%%%%%
function out = bilin(array, xy, nRGB,RGBsub)

if nargin < 3 || isempty(nRGB), nRGB = size(array,3); end;
if nargin < 4 || isempty(RGBsub)
    RGBsub = [];
    for RGB = 1:nRGB
        RGBsub = cat(1, RGBsub,ones(size(xy,1),1)*RGB);
    end
end

if nRGB ~= size(array,3) || length(RGBsub) ~= size(xy,1)*nRGB
    error('bilin: something''s up with your specification of RGB channels');
end

array = [array zeros(size(array,1),1,nRGB)];
array = [array; zeros(1,size(array,2),nRGB)];

xy = repmat(xy, [nRGB 1]);

ur = [ceil(xy(:,1)) floor(xy(:,2))];
ul = [floor(xy(:,1)) floor(xy(:,2))];
br = [ceil(xy(:,1)) ceil(xy(:,2))];
bl = [floor(xy(:,1)) ceil(xy(:,2))];

bad = br(:,1) == ul(:,1);
ur(bad,1) = ur(bad,1) + 1;
br(bad,1) = br(bad,1) + 1;

bad = bl(:,2) == ul(:,2);
br(bad,2) = br(bad,2) + 1;
bl(bad,2) = bl(bad,2) + 1;


indbl = sub2ind(size(array),bl(:,2),bl(:,1),RGBsub);
indbr = sub2ind(size(array),br(:,2),br(:,1),RGBsub);
indul = sub2ind(size(array),ul(:,2),ul(:,1),RGBsub);
indur = sub2ind(size(array),ur(:,2),ur(:,1),RGBsub);

vecarray = array(:);

x1out = ( ( (br(:,1)-xy(:,1))./(br(:,1)-bl(:,1)) ) .* vecarray(indbl) ) + ( ( (xy(:,1)-bl(:,1))./(br(:,1)-bl(:,1)) ) .* vecarray(indbr) );
x2out = ( ( (br(:,1)-xy(:,1))./(br(:,1)-bl(:,1)) ) .* vecarray(indul) ) + ( ( (xy(:,1)-bl(:,1))./(br(:,1)-bl(:,1)) ) .* vecarray(indur) );

out = ( ( (ul(:,2)-xy(:,2))./(ul(:,2)-bl(:,2)) ) .* x1out ) + ( ( (xy(:,2)-bl(:,2))./(ul(:,2)-bl(:,2)) ) .* x2out );
