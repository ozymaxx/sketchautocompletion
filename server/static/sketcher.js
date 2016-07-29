/**
 * Created by tugrulz on 25.07.2016.
 */

// Modify JSON.stringify to allow recursive and single-level arrays
var convArrToObj = function(array){
    var thisEleObj = new Object();
    if(typeof array == "object"){
        for(var i in array){
            var thisEle = convArrToObj(array[i]);
            thisEleObj[i] = thisEle;
        }
    }else {
        thisEleObj = array;
    }
    return thisEleObj;
}


var Point = function (x, y, timestamp, pid) {
    // String
    this.pid = "" + pid;
    // Long
    this.timestamp = timestamp;
    // Doubles
    this.x = x;
    this.y = y;

};



/*var Point1 = new Point(10.5, 9.4223, 1023);
console.log(Point1.toJSON());
Point1.jsonString();*/

var Stroke = function (width) {
    this.id = 0;
    this.points = [];
    //this.sid = Date.now() + "";
    //this.width = width;


};

var Stroke = function (width, id) {

    this.id = id;
    this.points = [];
    //this.sid = Date.now() + "";
    //this.width = width;


};

//noinspection JSAnnotator
Stroke.prototype.addPoint = function(x,y,timestamp, pid){
    this.points.push(new Point(x,y,timestamp, pid));
    //console.log("Pointler" + points.toString());
};

var Sketch = function () {

    this.strokes = [];
    this.skid = Date.now() + "";
    this.curStroke = undefined;
    //curStroke = new Stroke(width);

};


Sketch.prototype.newStroke = function(width, id) {
    this.curStroke = new Stroke(width, id);
    if (typeof this.curStroke === "undefined") {
        alert("something is undefined");
    }
    console.log("deneme" + this.curStroke);
    this.strokes.push(this.curStroke);
};

Sketch.prototype.addStroke = function( stroke ) {
    this.strokes.push(stroke);
};

Sketch.prototype.addPoint = function(x,y,timestamp, pid) {
    this.curStroke.addPoint(x,y,timestamp, pid);
};

Point.prototype.getJSON = function() {
    return {"pid":this.pid, "time":this.timestamp, "x":this.x,"y":this.y};
};


Stroke.prototype.getJSON = function() {

    /*this.jsonArr=[];
    for (i = 0; i < this.points.length; i++) {
        this.jsonArr.push(JSON.stringify(this.points[i].getJSON()));
        //console.log(jsonArr[i]);
    }*/

    return {"id":this.id, "points":this.points};
    //return {"id":this.id, "points":arrayJsonify(this.jsonArr)};



    /*this.jsonArr = [];


    for (i = 0; i < this.points.length; i++) {
        this.jsonArr.push(JSON.stringify(this.points[i].getJSON()));
        //console.log(jsonArr[i]);
    }
    //console.log("Point arrayi: " + JSON.stringify(this.jsonArr)));
    console.log("Stroke content: " + JSON.stringify({"id":this.id, "points":JSON.stringify(this.jsonArr)}));
    return {"id":this.id, "points":arrayJsonify(this.jsonArr)};*/
};







Sketch.prototype.getJSON = function() {
    return {"id":"candidate", "strokes":this.strokes};
    /*jsonArr = [];
    jsonArr[0] = "";
    for (i = 0; i < this.strokes.length; i++) {
        jsonArr.push(this.strokes[i].getJSON());
    }
    console.log(JSON.stringify(convArrToObj(this.jsonArr)));*/
    //return {"id":"candidate", "strokes":JSON.stringify(this.jsonArr)};
};



Sketch.prototype.getFinalResult = function() {
    return this.getJSON();
    //return JSON.stringify(this.strokes[0].getJSON());
    // return this.getJSON();
};

/* DUMP
Sketch.prototype.getJSONString = function() {
    return JSON.stringify(this.strokes[0].getJSON());
    return JSON.stringify();

    //return JSON.stringify(this.getJSON());
};

Stroke.prototype.getPointJSON = function() {
    console.log(this.points[0]);
    var po =  this.points[0];
    return po;
    //return JSON.stringify(po.getJSON());
};


Stroke.prototype.getJSONArray = function () {
    this.jsonArr=[];
    for (i = 0; i < this.points.length; i++) {
        this.jsonArr.push(JSON.stringify(this.points[i].getJSON()));
    }
    return this.jsonArr;

};

function arrayJsonify(jsonArr) {
    result = "[";
    for (i = 0; i < this.points.length; i++) {
        console.log("next to be inserted" + jsonArr[i]);
        result += jsonArr[i] + ",";
    }
    result += "]";
    return result;
};

Point.prototype.jsonString = function() {
    //console.log(this.getJSON());
};


 */
