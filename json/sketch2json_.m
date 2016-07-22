%{
{"id":"asdasd",
 "strokes":
    [{"id":"1","points":[{"pid":"qwe","time":4.6,"x":1,"y":9},{"pid":"aty","time":8,"x":6,"y":5}]},
    {"id":"21","points":[{"pid":"uio","time":9,"x":10.2,"y":44.4},{"pid":"asd","time":12,"x":3,"y":77},{"pid":"cvb","time":13,"x":5,"y":7}]
    }]
}
%}
counter = 1;
lastLabel = '';
for sketchid = 1:size(sketch,2)
    currSketch = sketch{sketchid}; % Strokes
    
    currLabel = char(labels(sketchid,1));
    
    if strcmp(currLabel, lastLabel)
        counter = counter + 1;
    else
        counter = 1;
        mkdir(strcat('./jsonfile/', currLabel , '/'));
    end
    lastLabel = currLabel;
    
    for partialid = 1:size(currSketch,2)-1
        sketchname = strcat(currLabel, '_', int2str(counter), '_' , int2str(partialid));
        filename = strcat('./jsonfile/', currLabel , '/', sketchname, '.json');
        fileID = fopen(filename, 'w');
        fprintf(fileID,'{"id":"'); 
        fprintf(fileID,'%s', sketchname); 
        fprintf(fileID, '",');   
        fprintf(fileID, '"strokes":['); 
        
        for partialstrokeid = 1:partialid
            currStroke = currSketch{partialstrokeid};
            fprintf(fileID, '{"id":"%i", "points":[', partialstrokeid); 
            for partialpointid = 1:size(currStroke,1)
                
                if partialpointid ~= size(currSketch{partialstrokeid},1)
                    fprintf(fileID, '{"pid":"%i", "time":1, "x":%.4f,"y":%.4f},', partialpointid, currStroke(partialpointid,1), currStroke(partialpointid,2));
                else
                   fprintf(fileID, '{"pid":"%i", "time":1, "x":%.4f,"y":%.4f}', partialpointid, currStroke(partialpointid,1), currStroke(partialpointid,2));
                end                
            
            end
            if partialstrokeid ~= partialid
                fprintf(fileID, ']},');
            else
               fprintf(fileID, ']}');
            end
            
        end
        fprintf(fileID,']}');
        fclose(fileID);
    end
end