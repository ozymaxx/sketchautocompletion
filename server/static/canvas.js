

/**
 * Created by tugrulz on 25.07.2016.
 */



// ISSUE: DO WE NEED TOUCH TOLERANCE?
console.log("BEN YAŞIYORUM ÖLMEDİM YANİ");

var canvas = document.getElementById("canvas");

var ctx = canvas.getContext('2d'),
    points = [],
    isDown = false,
    prevX, prevY;

width = 4.0;

var id = 0;
var pid = 0;

var sketch = new Sketch();

canvas.onmousedown = function(e) {

    console.log("BASTIN BANA");

    id++;
    pid++;
    console.log("Yeni point pidi " + pid);
    sketch.newStroke(width, id);

    /// adjust mouse position (see below)
    var pos = getXY(e);

    /// this is used to draw a line
    prevX = pos.x;
    prevY = pos.y;

    /// add new stroke
    //points.push([]);

    /// record point in this stroke
    //points[points.length - 1].push([pos.x, pos.y]);
    points.push([pos.x, pos.y]);

    sketch.addPoint(pos.x, pos.y, Date.now(), pid);

    /// we are in draw mode
    isDown = true;



};

canvas.onmousemove = function(e) {

    if (!isDown) return;

    var pos = getXY(e);

    /// draw a line from previous point to this
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    /// set previous to this point
    prevX = pos.x;
    prevY = pos.y;

    /// record to current stroke
    //points[points.length - 1].push([pos.x, pos.y]);
    points.push([pos.x, pos.y]);
    pid++;
    console.log("Yeni point pidi " + pid);
    sketch.addPoint(pos.x, pos.y, Date.now(), pid);
};

canvas.onmouseup = function() {
    isDown = false;
    //coords.innerHTML = JSON.stringify(points);
    coords.innerHTML = JSON.stringify(sketch.getFinalResult());
    pid = 0;

};

function getXY(e) {
    var r = canvas.getBoundingClientRect();
    return {x: e.clientX - r.left, y: e.clientY - r.top}
};

//render.onclick = function() {renderPoints(points);};
render.onclick = function() {clear();};


url = "http://localhost:5000";

post.onclick = function() {
    console.log("im onclick");
    send(url);
};

function renderPoints(points) {

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#f00';

    /// get a stroke
    console.log("Yeni" + points);
    for(var i = 0, t, p, pts; pts = points[i]; i++) {

        /// render stroke
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for(t = 1; p =pts[t]; t++) {
            ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
    }
    ctx.strokeStyle = '#000';
};

function clear() {
    sketch.newStroke(width);
    points = [];
    coords.innerHTML = "";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    id = 0;
    pid = 0;
    console.log("Yeni pointlerin de pidi " + pid);
};


function send(URL) {
    console.log("sending to " + URL);
    $.post(URL,JSON.stringify(points));
    clear();
};

$(document).ready(function(){
    $("button").click(function(){
        $("#div1").load("demo_test.txt");
    });
});