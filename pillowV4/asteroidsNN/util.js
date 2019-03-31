
function sq(a) { return a ** 2; }

function intersection(r, h, k, [m, n]) {
    // circle: (x - h)^2 + (y - k)^2 = r^2
    // line: y = m * x + n
    // r: circle radius
    // h: x value of circle centre
    // k: y value of circle centre
    // m: slope
    // n: y-intercept

    // get a, b, c values
    var a = 1 + sq(m);
    var b = -h * 2 + (m * (n - k)) * 2;
    var c = sq(h) + sq(n - k) - sq(r);

    // get discriminant
    var d = sq(b) - 4 * a * c;
    if (d >= 0) {
        // insert into quadratic formula
        var intersections = [
            (-b + Math.sqrt(sq(b) - 4 * a * c)) / (2 * a),
            (-b - Math.sqrt(sq(b) - 4 * a * c)) / (2 * a)
        ];
        return intersections;
    }
    // no intersection
    return [];
}
function getEquationOfLineFromTwoPoints(x1, y1, x2, y2) {
    //if(Math.abs(y1 - y2) < 0.001) {
    //}
    return [ (y1 - y2) / (x1 - x2), (x1*y2-x2*y1) / (x1-x2)];
}
