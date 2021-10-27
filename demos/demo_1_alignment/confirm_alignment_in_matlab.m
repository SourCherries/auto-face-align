close all; clearvars; clc;

folders = {'faces', 'aligned'};
fprintf(1, '\n%s\n%s\n', '*****************************************', 'Options:')
fprintf(1, '\t%1d\t%s\n', 1, 'Original faces')
fprintf(1, '\t%1d\t%s\n', 2, 'Aligned faces')
folderi = input('Enter option number: ');

data = jsondecode(fileread([folders{folderi}, '/landmarks.txt']));
guys = fieldnames(data);

if folderi == 2
    % get size & initialize matrix to store all faces
    parts = regexp(guys{1},'_','split');
    filename = [folders{folderi}, '/', parts{1}, '/', parts{2}, '-', parts{3}, '.', parts{4}];
    face = imread(filename);
    [rows, cols, channels] = size(face);
    faces = zeros(rows, cols, channels, length(guys));
end

for i = 1 : length(guys)
    poo = data.(guys{i});
    
    features = fieldnames(poo);
    X = []; Y = [];    
    for fi = 1 : length(features)
        eval(['X = [X; poo.', features{fi}, '(1:2:end)];'])
        eval(['Y = [Y; poo.', features{fi}, '(2:2:end)];'])
    end
    
    parts = regexp(guys{i},'_','split');
    filename = [folders{folderi}, '/', parts{1}, '/', parts{2}, '-', parts{3}, '.', parts{4}];
    face = imread(filename);
    
    figure; imshow(face,[]); hold on;
    plot(X, Y, 'r.', 'MarkerSize', 12);
    
    if folderi == 2
        faces(:, :, :, i) = face;
    end
end

if folderi == 2
    MF = mean(faces, 4);
    MF = MF - min(MF(:));
    MF = uint8(MF * 255 / max(MF(:)));
    figure; imshow(MF);
end