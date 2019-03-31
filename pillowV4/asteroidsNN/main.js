const canvas = document.getElementById('canvas');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const ctx = canvas.getContext('2d');

function rand() {
    return Math.random() * 0.8 + 0.1;
}

var spaceships = Array(1).fill(null).map(() => {
    return new SpaceShip(canvas.width * 0.05, canvas.height * rand(), ctx);
})

const OBSTACLE_COUNT = 200;

var obstacles = Array(OBSTACLE_COUNT).fill(null).map(() => 
    new Obstacle(canvas.width * rand(), canvas.height * Math.random(), ctx))

var heldKeys = [];
window.addEventListener('keydown', event => {
    console.log(event)
    if(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
        if(heldKeys.includes(event.key))
            return;
        heldKeys.push(event.key)
    }
}, true)
window.addEventListener('keyup', event => {
    console.log(event)
    if(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
        if(! heldKeys.includes(event.key))
            return;
        var index = heldKeys.indexOf(event.key);
        if (index !== -1) heldKeys.splice(index, 1);
    }
}, true)

setInterval(() => {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if(heldKeys.includes('ArrowLeft')) {
        spaceships.forEach(ship => { ship.onTurnLeft() })
    } else if(heldKeys.includes('ArrowRight')) {
        spaceships.forEach(ship => { ship.onTurnRight() })
    } else if(heldKeys.includes('ArrowUp')) {
        spaceships.forEach(ship => { ship.onSpeedUp() })
    } else if(heldKeys.includes('ArrowDown')) {
        spaceships.forEach(ship => { ship.onSpeedDown() })
    }
    spaceships.forEach(ship => {
        ship.proceed()
        ship.draw()
        ship.radar(obstacles)
        if(ship.checkCollisions(obstacles))
            window.location = window.location;
    })
    obstacles.forEach(obst => {
        obst.draw()
    })
}, 1000/60);
