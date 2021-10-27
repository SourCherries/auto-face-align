close all; clearvars; clc;

data = jsondecode(fileread('aligned/landmarks.txt'));
guys = fieldnames(data);
nfaces = length(guys);


% get size
parts = regexp(guys{1},'_','split');
filename = ['aligned/', parts{2}, '/', parts{3}, '-', parts{4}, '.', parts{5}];
face = imread(filename);
[rows, cols, channels] = size(face);

[X,Y] = deal(cell(nfaces,1));
faces = zeros(rows, cols, channels, length(guys));
for i = 1 : nfaces
    poo = data.(guys{i});
    
    features = fieldnames(poo);
    X{i} = []; Y{i} = [];    
    for fi = 1 : length(features)
        X{i} = [X{i}; poo.(features{fi})(1:2:end)];
        Y{i} = [Y{i}; poo.(features{fi})(2:2:end)];
    end
end

npoints = length(X{1});
[mnX,mnY] = deal(zeros(npoints,1));

for i = 1:nfaces
    mnX = mnX + X{i}./nfaces;
    mnY = mnY + Y{i}./nfaces;
end

for i = 1:nfaces
    parts = regexp(guys{i},'_','split');
    filename = [parts{3}, '-', parts{4}, '.', parts{5}];
    inpth1 = 'aligned';
    inpth2 = parts{2};
    fullfilename = [inpth1,'/',inpth2,'/',filename];
    face = imread(fullfilename);
    warpface = pawarp(face,[mnX,mnY],[X{i},Y{i}], 'fullim','yes');
    figure; imshow(warpface); hold on;
    outpth1 = 'warped';
    outpth2 = parts{2};
    fulloutpth = [outpth1,'/',outpth2];
    if ~exist(fulloutpth,'dir')
        mkdir(fulloutpth);
    end
    fulloutname = [fulloutpth,'/',filename];
    imwrite(warpface,fulloutname,'quality',100);
end
