class SpaceShip {
    
    constructor(x, y, ctx) {
        this.x = x;
        this.y = y;
        this.ctx = ctx;
        this.angle = 0;
        this.r = 10;
        this.velocity = 0;
        this.rotationVelocity = 0;
    }

    draw() {
        this.ctx.fillStyle = "red";
        this.ctx.beginPath();
        this.ctx.moveTo(this.x, this.y);

        for(var i = 0; i < 4; i++) {
            let corner = movePoint(this.x, this.y, this.angle + (i%3)*360/3, this.r);

            this.ctx[i == 0 ? 'moveTo' : 'lineTo'](corner[0], corner[1]);       
        }
        this.ctx.fill();
    }

    proceed() {
        var newPosition = movePoint(this.x, this.y, this.angle, this.velocity);

        this.x = newPosition[0];
        this.y = newPosition[1];

        this.angle += this.rotationVelocity;

        if(Math.abs(this.rotationVelocity) < 0.15) {
            this.rotationVelocity = 0;
        } else if(this.rotationVelocity > 0) {
            this.rotationVelocity -= 0.2;
        } else {
            this.rotationVelocity += 0.2;
        }
    } 

    checkCollisions(obstacles) {
        let output = false;
        let corner = []

        for(var i = 0; i < 3; i++) {
            corner[i] = movePoint(this.x, this.y, this.angle + i*360/3, this.r);
        }

        obstacles.forEach(obstacle => { 
           for(var i = 0; i < 3; i++) {
                if(dist(obstacle.x, obstacle.y, corner[i][0], corner[i][1]) < obstacle.r) {
                    output = true;
                }
            }        
        })
        return output;
    }   

    onTurnLeft() {
        this.rotationVelocity -= this.rotationVelocity > -1 ? 0.5 : 0.3;
    }
    
    onTurnRight() {
        this.rotationVelocity += this.rotationVelocity < 1 ? 0.5 : 0.3;
    }

    onSpeedUp() {
        this.velocity += 0.1;
    }

    onSpeedDown() {
        this.velocity -= 0.2;
        
        this.velocity = this.velocity < 0 ? 0 : this.velocity;
    }


    radar(obstacles) {
        var lines = []            
        var distances = Array(10).fill(200);

        this.ctx.strokeStyle = "#ff0000";
        for(var i = 0; i < 10; i++) {
            var movedRadarPt = movePoint(this.x, this.y, this.angle + i*36, 200)            

            this.ctx.beginPath();
            this.ctx.moveTo(this.x, this.y);
            this.ctx.lineTo(movedRadarPt[0], movedRadarPt[1]);
            this.ctx.stroke();
            this.ctx.strokeStyle = "#00ff00";
    
            lines.push(getEquationOfLineFromTwoPoints(this.x, this.y, movedRadarPt[0], movedRadarPt[1]))
        }

        obstacles.forEach(obstacle => { 
            for(var i = 0; i < 10; i++) {
                let intersectionPoints = intersection(obstacle.x, obstacle.y, obstacle.r, lines[i]);

                if(intersectionPoints.length == 0) continue;

                let d = Math.min(dist(this.x, this.y, intersectionPoints[0][0], intersectionPoints[0][1]), 
                           dist(this.x, this.y, intersectionPoints[1][0], intersectionPoints[1][1]));

                if(d < distances[i]) {
                    distances[i] = d;
                }
            }
        })
        return distances;
    }

   }

function movePoint(x, y, a, R) {
    return [x + Math.cos(a * Math.PI / 180.0) * R, y + Math.sin(a * Math.PI / 180.0) * R];
}

function dist(ax, ay, bx, by) {
    return Math.sqrt((ax - bx)**2 + (ay - by)**2);
}
