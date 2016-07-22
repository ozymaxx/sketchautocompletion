%{
{"id":"asdasd",
 "strokes":
    [{"id":"1","points":[{"pid":"qwe","time":4.6,"x":1,"y":9},{"pid":"aty","time":8,"x":6,"y":5}]},
    {"id":"21","points":[{"pid":"uio","time":9,"x":10.2,"y":44.4},{"pid":"asd","time":12,"x":3,"y":77},{"pid":"cvb","time":13,"x":5,"y":7}]
    }]
}
%}
%for sketchid = 1:size(sketch,2)
counter = 1;
lastLabel = '';
for sketchid = 1:size(sketch,2)
    currLabel = char(labels(sketchid,1));
    
    if strcmp(currLabel, lastLabel)
        counter = counter + 1;
    else
        counter = 1;
        mkdir(strcat('./jsonfile/', currLabel , '/'));
    end
    lastLabel = currLabel;
    
    currSketch = sketch{sketchid}; % Strokes
    for strokeid = 1:size(currSketch,2)
        for partialid = 1:strokeid-1
            sketchname = strcat(currLabel, '_', int2str(counter), '_' , int2str(partialid));
            filename = strcat('./jsonfile/', currLabel , '/', sketchname, '.json');
            fileID = fopen(filename, 'w');
            fprintf(fileID,'{"id":"'); 
            fprintf(fileID,'%s', sketchname); 
            fprintf(fileID, '",');   
            fprintf(fileID, '"strokes":[');

        currStroke = currSketch{strokeid};
        fprintf(fileID, '{"id":"%i", "points":[', strokeid);    
            
                if pointId ~= size(currStroke,1)
                    fprintf(fileID, '{"pid":"%i", "time":1, "x":%.4f,"y":%.4f},', pointId, currStroke(pointId,1), currStroke(pointId,2));
                else
                   fprintf(fileID, '{"pid":"%i", "time":1, "x":%.4f,"y":%.4f}', pointId, currStroke(pointId,1), currStroke(pointId,2));
                end
            end
        end
        if strokeid ~= size(currSketch,2)
            fprintf(fileID, ']},');
        else
           fprintf(fileID, ']}');
        end
        fclose(fileID);
    end
    fprintf(fileID,']}');
   
end