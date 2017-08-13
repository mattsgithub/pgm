var drag_line = undefined;
var nodes = [];
var svg = undefined;

function dragmove() {
    drag_line.attr("x2", d3.event.x);
    drag_line.attr("y2", d3.event.y);
};

function dragstarted() {
    drag_line.style("opacity", 1.0)
    drag_line.attr("x1", d3.event.x);
    drag_line.attr("y1", d3.event.y);
    drag_line.attr("x2", d3.event.x);
    drag_line.attr("y2", d3.event.y);
};

function intersection(x0, y0, x, y, r) {
    d = (x - x0)**2 + (y - y0)**2
    return d <= r**2
};

function dragended() {
    drag_line.style("opacity", 0.0)
    var x = d3.event.x;
    var y = d3.event.y;

    for (var i = 0; i < nodes.length; i++) {
        if (intersection(nodes[i].x, nodes[i].y, x, y, 45.0)) {
            var x1 = drag_line.attr("x1");
            var y1 = drag_line.attr("y1");
            svg.append("line")
                    .attr("stroke", "black")
                    .attr("marker-end", "url(#arrow)")
                    .attr("stroke-width", 5)
                    .attr("x1", x1)
                    .attr("y1", y1)
                    .attr("x2", d3.event.x)
                    .attr("y2", d3.event.y);
        }
    }
};

window.onload = function() {
    svg = d3.select("body").append("svg");
    svg.attr("width", 960)
    svg.attr("height", 960);

    var defs = svg.append("defs");
    var arrow_marker = defs.append("marker");
    arrow_marker.attr("id", "arrow")
       .attr("orient", "auto")
       .attr("refX", 0)
       .attr("refY", 3)
       .attr("markerUnits", "strokeWidth")
       .attr("markerHeight", 10)
       .attr("markerWidth", 10);
    arrow_marker.append("path").attr("d", "M 0 0 L 0 6 L 9 3");

    drag_line = svg.append("line")
                    .attr("class", "hidden")
                    .style("opacity", 0.0)
                    .attr("stroke", "black")
                    .attr("marker-end", "url(#arrow)")
                    .attr("stroke-width", 5);

    var drag = d3.drag();
    drag.on("start", dragstarted);
    drag.on("drag", dragmove);
    drag.on("end", dragended);

    svg.on("click", function() {
        var coords = d3.mouse(this);
        nodes.push({'r': 50.0,
                    'x': coords[0],
                    'y': coords[1]});

        svg.selectAll("circle")
            .data(nodes)
            .enter()
            .append("circle")
            .attr("cx", function(d) {return d.x})
            .attr("cy", function(d) {return d.y})
            .attr("r", function(d) {return d.r})
            .style("fill", "#c4c4c4")
            .call(drag);
    });
};
