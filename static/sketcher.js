/**
 * Created by tugrulz on 25.07.2016.
 */



var Point = function (x, y, timestamp) {
    // Doubles
    this.x = x;
    this.y = y;

    // Long
    this.timestamp = timestamp;

    // String
    this.pid = "" + timestamp;
};

Point.prototype.jsonString = function() {
    console.log(this.toJSON());
};

/*var Point1 = new Point(10.5, 9.4223, 1023);
console.log(Point1.toJSON());
Point1.jsonString();*/

var Stroke = function (width) {

    this.points = [];
    this.sid = Date.now() + "";
    this.width = width;

};

//noinspection JSAnnotator
Stroke.prototype.addPoint = function(x,y,timestamp){
    points.push(new Point(x,y,timestamp));
};

var Sketch = function () {

    this.strokes = [];
    this.skid = Date.now() + "";
    this.curStroke = undefined;
    //curStroke = new Stroke(width);

};


Sketch.prototype.newStroke = function(width) {
    this.curStroke = new Stroke(width);
    if (typeof this.curStroke === "undefined") {
        alert("something is undefined");
    }
    console.log("deneme" + this.curStroke);
    this.strokes.push(this.curStroke);
};

Sketch.prototype.addStroke = function( stroke ) {
    this.strokes.add(stroke);
};

Sketch.prototype.addPoint = function(x,y,timestamp) {
    this.curStroke.addPoint(x,y,timestamp);
};
