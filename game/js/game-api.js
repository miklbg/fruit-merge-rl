/**
 * Fruit Merge Game API
 * 
 * A modular API for the Fruit Merge game that supports:
 * - Normal gameplay with rendering
 * - Headless mode for AI training (no rendering)
 * - Fast physics mode (run Matter.js as fast as possible)
 * 
 * @module game-api
 */

// Game configuration constants (matching original game)
const NATIVE_SCALE = 3;
const NATIVE_WIDTH = 400 * NATIVE_SCALE;
const NATIVE_HEIGHT = 600 * NATIVE_SCALE;

const FRUITS = [
    { level: 0, baseRadius: 22 * NATIVE_SCALE, radius: 21.00 * NATIVE_SCALE, offsetX: 0, offsetY: 1.1 * NATIVE_SCALE, color: '#2E3771', imagePath: 'assets/1-blueberry.png', score: 5 },
    { level: 1, baseRadius: 26 * NATIVE_SCALE, radius: 22.80 * NATIVE_SCALE, offsetX: -0.5, offsetY: 2.6 * NATIVE_SCALE, color: '#842F28', imagePath: 'assets/2-strawberry.png', score: 10 },
    { level: 2, baseRadius: 34 * NATIVE_SCALE, radius: 32.00 * NATIVE_SCALE, offsetX: 0, offsetY: 1 * NATIVE_SCALE, color: '#612B53', imagePath: 'assets/6-grapes.png', score: 20 },
    { level: 3, baseRadius: 38 * NATIVE_SCALE, radius: 36.00 * NATIVE_SCALE, offsetX: 0, offsetY: 2 * NATIVE_SCALE, color: '#B06322', imagePath: 'assets/5-orange.png', score: 35 },
    { level: 4, baseRadius: 48 * NATIVE_SCALE, radius: 45.60 * NATIVE_SCALE, offsetX: 0, offsetY: 6 * NATIVE_SCALE, color: '#952C29', imagePath: 'assets/4-apple.png', score: 55 },
    { level: 5, baseRadius: 58 * NATIVE_SCALE, radius: 52.00 * NATIVE_SCALE, color: '#AE993B', imagePath: 'assets/3-lemon.png', score: 80 },
    { level: 6, baseRadius: 66 * NATIVE_SCALE, radius: 60.00 * NATIVE_SCALE, offsetX: 0, offsetY: 6 * NATIVE_SCALE, color: '#8B8A62', imagePath: 'assets/7-cantaloupe.png', score: 110 },
    { level: 7, baseRadius: 78 * NATIVE_SCALE, radius: 70.00 * NATIVE_SCALE, offsetX: 0, offsetY: 8 * NATIVE_SCALE, color: '#A27620', imagePath: 'assets/9-pineapple.png', score: 150 },
    { level: 8, baseRadius: 90 * NATIVE_SCALE, radius: 77.00 * NATIVE_SCALE, offsetX: 0, offsetY: 12 * NATIVE_SCALE, color: '#67412B', imagePath: 'assets/8-coconut.png', score: 200 },
    { level: 9, baseRadius: 102 * NATIVE_SCALE, radius: 94.00 * NATIVE_SCALE, offsetX: 0, offsetY: 8 * NATIVE_SCALE, color: '#5B723A', imagePath: 'assets/10-watermelon.png', score: 300 }
];

const MAX_FRUIT_LEVEL = FRUITS.length - 1;
const STARTING_FRUIT_LEVELS = 4;
const GAME_OVER_LINE_Y_PERCENT = 0.18;
const DROP_AREA_Y_PERCENT = 0.1;
const BASE_WALL_THICKNESS = 40 * NATIVE_SCALE;
const GROUND_HEIGHT = 48 * NATIVE_SCALE;
const DROP_COOLDOWN_MS = 400;

/**
 * Creates a new Fruit Merge game instance
 * 
 * @param {Object} options - Configuration options
 * @param {boolean} [options.headless=false] - If true, disables rendering for faster simulation
 * @param {boolean} [options.fastMode=false] - If true, runs physics as fast as possible
 * @param {HTMLElement} [options.container=null] - Container element for rendering (required if not headless)
 * @param {Object} [options.Matter=null] - Matter.js module reference (required)
 * @returns {Object} Game API object
 */
export function createGame(options = {}) {
    const {
        headless = false,
        fastMode = false,
        container = null,
        Matter = null
    } = options;

    if (!Matter) {
        throw new Error('Matter.js module is required. Pass it via options.Matter');
    }

    // Matter.js modules
    const { Engine, Render, Runner, World, Bodies, Body, Events, Composite, Common } = Matter;

    // Game dimensions (fixed size)
    const gameWorldWidth = NATIVE_WIDTH;
    const gameWorldHeight = NATIVE_HEIGHT;
    const gameOverLineY = gameWorldHeight * GAME_OVER_LINE_Y_PERCENT;
    const dropAreaY = gameWorldHeight * DROP_AREA_Y_PERCENT;
    const wallThickness = BASE_WALL_THICKNESS;

    // Game state
    let engine = null;
    let render = null;
    let runner = null;
    let world = null;
    let score = 0;
    let currentFruitLevel = 0;
    let nextFruitLevel = 0;
    let isGameOver = false;
    let isDropping = false;
    let lastDropTime = 0;
    let stepCount = 0;
    let mergeCount = 0;
    let maxFruitLevel = 0;

    // Bodies pending merge processing
    let bodiesToRemoveSet = new Set();
    let bodiesToAddArray = [];

    // Event callbacks
    let onMerge = null;
    let onGameOver = null;
    let onScoreUpdate = null;
    let onDrop = null;

    // Fast mode loop control
    let fastModeRunning = false;
    let fastModeAnimationFrame = null;

    /**
     * Converts hex color to rgba
     */
    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    /**
     * Creates a fruit body at the specified position
     */
    function createFruitBody(x, y, level, bodyOptions = {}) {
        const fruitData = FRUITS[level];
        if (!fruitData) return null;

        const options = {
            label: 'fruit',
            fruitLevel: level,
            friction: 0.3,
            restitution: 0.2,
            density: 0.001,
            render: {
                fillStyle: hexToRgba(fruitData.color, headless ? 0 : 0.1),
                strokeStyle: 'transparent',
                lineWidth: 0
            },
            ...bodyOptions
        };

        const radius = fruitData.radius !== undefined ? fruitData.radius : fruitData.baseRadius;
        const offsetX = fruitData.offsetX || 0;
        const offsetY = fruitData.offsetY || 0;

        const bodyX = x + offsetX;
        const bodyY = y + offsetY;

        return Bodies.circle(bodyX, bodyY, radius, options);
    }

    /**
     * Spawns the next random fruit for preview
     */
    function spawnNextFruit() {
        nextFruitLevel = Math.floor(Math.random() * STARTING_FRUIT_LEVELS);
    }

    /**
     * Sets current fruit to next fruit and spawns new next
     */
    function advanceFruit() {
        currentFruitLevel = nextFruitLevel;
        spawnNextFruit();
    }

    /**
     * Processes collision events for merging
     */
    function handleCollisionStart(event) {
        if (isGameOver) return;

        const pairs = event.pairs;
        for (let i = 0; i < pairs.length; i++) {
            const pair = pairs[i];
            const bodyA = pair.bodyA;
            const bodyB = pair.bodyB;

            if (bodyA.label !== 'fruit' || bodyB.label !== 'fruit') continue;
            if (bodiesToRemoveSet.has(bodyA.id) || bodiesToRemoveSet.has(bodyB.id)) continue;

            if (bodyA.fruitLevel === bodyB.fruitLevel) {
                const level = bodyA.fruitLevel;
                if (level === MAX_FRUIT_LEVEL) continue;

                // Prevent merging if both fruits are above the game over line
                const bodyAAboveLine = bodyA.position.y < gameOverLineY;
                const bodyBAboveLine = bodyB.position.y < gameOverLineY;
                if (bodyAAboveLine && bodyBAboveLine) continue;

                const newX = (bodyA.position.x + bodyB.position.x) / 2;
                const newY = (bodyA.position.y + bodyB.position.y) / 2;
                const nextLevel = level + 1;

                bodiesToAddArray.push({
                    x: newX,
                    y: newY,
                    level: nextLevel,
                    score: FRUITS[level].score
                });

                bodiesToRemoveSet.add(bodyA.id);
                bodiesToRemoveSet.add(bodyB.id);
            }
        }
    }

    /**
     * Processes pending merges after physics update
     */
    function handleAfterUpdate() {
        if (bodiesToRemoveSet.size > 0) {
            const bodiesToRemove = Composite.allBodies(world).filter(body => bodiesToRemoveSet.has(body.id));
            bodiesToRemove.forEach(body => {
                World.remove(world, body);
            });

            bodiesToAddArray.forEach(data => {
                const newFruit = createFruitBody(data.x, data.y, data.level);
                Body.setVelocity(newFruit, { x: Common.random(-1, 1), y: -2 });
                World.add(world, newFruit);

                score += data.score;
                mergeCount++;

                if (data.level > maxFruitLevel) {
                    maxFruitLevel = data.level;
                }

                if (onScoreUpdate) {
                    onScoreUpdate(score);
                }

                if (onMerge) {
                    onMerge({
                        newLevel: data.level,
                        position: { x: data.x, y: data.y },
                        scoreGained: data.score,
                        totalScore: score
                    });
                }
            });

            bodiesToRemoveSet.clear();
            bodiesToAddArray = [];
        }

        stepCount++;
    }

    /**
     * Checks for game over condition
     */
    function handleCollisionStay(event) {
        if (isGameOver) return;

        const pairs = event.pairs;
        for (let i = 0; i < pairs.length; i++) {
            const pair = pairs[i];
            const bodyA = pair.bodyA;
            const bodyB = pair.bodyB;
            let fruitBody = null;

            if (bodyA.label === 'gameOverLine' && bodyB.label === 'fruit') fruitBody = bodyB;
            if (bodyB.label === 'gameOverLine' && bodyA.label === 'fruit') fruitBody = bodyA;

            if (fruitBody) {
                const isCenterAboveLine = fruitBody.position.y < gameOverLineY;
                if (isCenterAboveLine && 
                    fruitBody.velocity.y < 0.05 && 
                    fruitBody.velocity.y > -0.05 && 
                    Math.abs(fruitBody.velocity.x) < 0.1) {
                    triggerGameOver();
                    break;
                }
            }
        }
    }

    /**
     * Triggers game over state
     */
    function triggerGameOver() {
        if (isGameOver) return;
        isGameOver = true;

        if (fastModeRunning) {
            stopFastMode();
        }

        if (onGameOver) {
            onGameOver({
                score,
                stepCount,
                mergeCount,
                maxFruitLevel
            });
        }
    }

    /**
     * Runs the fast mode loop using requestAnimationFrame
     */
    function runFastModeLoop() {
        if (!fastModeRunning || isGameOver) return;

        // Run multiple physics steps per frame for maximum speed
        const stepsPerFrame = 10;
        for (let i = 0; i < stepsPerFrame && !isGameOver; i++) {
            Engine.update(engine, 1000 / 60); // 60 FPS timestep
        }

        fastModeAnimationFrame = requestAnimationFrame(runFastModeLoop);
    }

    /**
     * Starts fast mode execution
     */
    function startFastMode() {
        if (fastModeRunning) return;
        fastModeRunning = true;

        if (runner) {
            Runner.stop(runner);
        }

        runFastModeLoop();
    }

    /**
     * Stops fast mode execution
     */
    function stopFastMode() {
        fastModeRunning = false;
        if (fastModeAnimationFrame) {
            cancelAnimationFrame(fastModeAnimationFrame);
            fastModeAnimationFrame = null;
        }
    }

    /**
     * Initializes the game engine and world
     */
    function init() {
        // Create engine
        engine = Engine.create();
        world = engine.world;
        world.gravity.y = 1.2;

        // Create renderer if not headless
        if (!headless && container) {
            render = Render.create({
                element: container,
                engine: engine,
                options: {
                    width: gameWorldWidth,
                    height: gameWorldHeight,
                    wireframes: false,
                    background: 'transparent'
                }
            });

            render.bounds.min.x = 0;
            render.bounds.min.y = 0;
            render.bounds.max.x = gameWorldWidth;
            render.bounds.max.y = gameWorldHeight;
        }

        // Create runner
        runner = Runner.create();

        // Create walls
        const wallOptions = {
            isStatic: true,
            restitution: 0.1,
            friction: 0.2,
            render: {
                fillStyle: 'transparent',
                strokeStyle: 'transparent',
                lineWidth: 0
            }
        };

        const wallHeight = gameWorldHeight - gameOverLineY;
        const wallCenterY = gameOverLineY + wallHeight / 2;

        World.add(world, [
            Bodies.rectangle(gameWorldWidth / 2, gameWorldHeight - GROUND_HEIGHT / 2, gameWorldWidth, GROUND_HEIGHT, { ...wallOptions, label: 'ground' }),
            Bodies.rectangle(wallThickness / 2, wallCenterY, wallThickness, wallHeight, { ...wallOptions, label: 'wall-left' }),
            Bodies.rectangle(gameWorldWidth - wallThickness / 2, wallCenterY, wallThickness, wallHeight, { ...wallOptions, label: 'wall-right' })
        ]);

        // Create game over line (sensor)
        const gameOverLine = Bodies.rectangle(
            gameWorldWidth / 2,
            gameOverLineY,
            gameWorldWidth,
            2,
            {
                isStatic: true,
                isSensor: true,
                label: 'gameOverLine',
                render: {
                    fillStyle: 'transparent',
                    strokeStyle: 'transparent',
                    lineWidth: 0
                }
            }
        );
        World.add(world, gameOverLine);

        // Setup event handlers
        Events.on(engine, 'collisionStart', handleCollisionStart);
        Events.on(engine, 'afterUpdate', handleAfterUpdate);
        Events.on(engine, 'collisionStay', handleCollisionStay);

        // Initialize fruits
        spawnNextFruit();
        advanceFruit();
        spawnNextFruit();

        // Reset state
        score = 0;
        isGameOver = false;
        isDropping = false;
        lastDropTime = 0;
        stepCount = 0;
        mergeCount = 0;
        maxFruitLevel = 0;
        bodiesToRemoveSet.clear();
        bodiesToAddArray = [];
    }

    /**
     * Starts the game loop
     */
    function start() {
        if (!engine) {
            init();
        }

        if (!headless && render) {
            Render.run(render);
        }

        if (fastMode) {
            startFastMode();
        } else {
            Runner.run(runner, engine);
        }
    }

    /**
     * Stops the game loop
     */
    function stop() {
        if (fastModeRunning) {
            stopFastMode();
        }

        if (runner) {
            Runner.stop(runner);
        }

        if (render) {
            Render.stop(render);
        }
    }

    /**
     * Resets the game to initial state
     */
    function reset() {
        stop();

        // Clear event handlers
        if (engine) {
            Events.off(engine);
        }

        // Clear world
        if (world) {
            World.clear(world, false);
            Composite.clear(world, false, true);
        }

        // Clear engine
        if (engine) {
            Engine.clear(engine);
        }

        // Remove canvas if rendered
        if (render && render.canvas) {
            render.canvas.remove();
        }

        // Reset references
        engine = null;
        render = null;
        runner = null;
        world = null;

        // Reinitialize
        init();
    }

    /**
     * Drops a fruit at the specified x position
     * 
     * @param {number} x - X position in game world coordinates (0 to NATIVE_WIDTH)
     * @returns {boolean} True if drop was successful, false otherwise
     */
    function dropFruit(x) {
        if (isGameOver || isDropping) return false;

        const currentTime = Date.now();
        if (!fastMode && currentTime - lastDropTime < DROP_COOLDOWN_MS) {
            return false;
        }

        // Constrain x position
        const fruitData = FRUITS[currentFruitLevel];
        const radius = fruitData.baseRadius;
        const constrainedX = Math.max(
            radius + wallThickness,
            Math.min(gameWorldWidth - radius - wallThickness, x)
        );

        // Create and drop fruit
        const droppedFruit = createFruitBody(constrainedX, dropAreaY, currentFruitLevel);
        if (!droppedFruit) return false;

        Body.setVelocity(droppedFruit, { x: 0, y: 10 });
        World.add(world, droppedFruit);

        lastDropTime = currentTime;
        isDropping = true;

        if (onDrop) {
            onDrop({
                level: currentFruitLevel,
                position: { x: constrainedX, y: dropAreaY }
            });
        }

        // Advance to next fruit after cooldown (or immediately in fast mode)
        if (fastMode) {
            advanceFruit();
            isDropping = false;
        } else {
            setTimeout(() => {
                if (!isGameOver) {
                    advanceFruit();
                    isDropping = false;
                }
            }, DROP_COOLDOWN_MS);
        }

        return true;
    }

    /**
     * Performs a single physics step (useful for training)
     * 
     * @param {number} [deltaTime=16.67] - Time step in milliseconds (default 60 FPS)
     */
    function step(deltaTime = 1000 / 60) {
        if (engine && !isGameOver) {
            Engine.update(engine, deltaTime);
        }
    }

    /**
     * Runs multiple physics steps at once
     * 
     * @param {number} count - Number of steps to run
     * @param {number} [deltaTime=16.67] - Time step per update in milliseconds
     */
    function stepMultiple(count, deltaTime = 1000 / 60) {
        for (let i = 0; i < count && !isGameOver; i++) {
            step(deltaTime);
        }
    }

    /**
     * Gets the current game state (useful for AI observation)
     * 
     * @returns {Object} Current game state
     */
    function getState() {
        const fruits = [];
        if (world) {
            Composite.allBodies(world).forEach(body => {
                if (body.label === 'fruit') {
                    fruits.push({
                        level: body.fruitLevel,
                        x: body.position.x,
                        y: body.position.y,
                        velocityX: body.velocity.x,
                        velocityY: body.velocity.y,
                        angle: body.angle,
                        angularVelocity: body.angularVelocity,
                        radius: FRUITS[body.fruitLevel].radius
                    });
                }
            });
        }

        return {
            score,
            isGameOver,
            isDropping,
            currentFruitLevel,
            nextFruitLevel,
            stepCount,
            mergeCount,
            maxFruitLevel,
            fruits,
            gameWidth: gameWorldWidth,
            gameHeight: gameWorldHeight,
            gameOverLineY,
            dropAreaY,
            wallThickness
        };
    }

    /**
     * Gets normalized state suitable for neural network input
     * 
     * @returns {Object} Normalized state with values between 0 and 1
     */
    function getNormalizedState() {
        const state = getState();
        
        // Normalize positions and velocities
        const normalizedFruits = state.fruits.map(fruit => ({
            level: fruit.level / MAX_FRUIT_LEVEL,
            x: fruit.x / gameWorldWidth,
            y: fruit.y / gameWorldHeight,
            velocityX: Math.tanh(fruit.velocityX / 10), // Normalize velocities
            velocityY: Math.tanh(fruit.velocityY / 10),
            angle: (fruit.angle % (2 * Math.PI)) / (2 * Math.PI),
            angularVelocity: Math.tanh(fruit.angularVelocity)
        }));

        return {
            score: state.score,
            normalizedScore: Math.tanh(state.score / 10000),
            isGameOver: state.isGameOver ? 1 : 0,
            currentFruitLevel: state.currentFruitLevel / MAX_FRUIT_LEVEL,
            nextFruitLevel: state.nextFruitLevel / MAX_FRUIT_LEVEL,
            fruits: normalizedFruits,
            fruitCount: normalizedFruits.length,
            maxFruitLevel: state.maxFruitLevel / MAX_FRUIT_LEVEL
        };
    }

    /**
     * Gets a simplified observation vector for RL training
     * 
     * @param {number} [maxFruits=20] - Maximum number of fruits to include
     * @returns {Float32Array} Flat array of observations
     */
    function getObservation(maxFruits = 20) {
        const state = getNormalizedState();
        
        // Features per fruit: level, x, y, velocityX, velocityY
        const featuresPerFruit = 5;
        const staticFeatures = 4; // currentFruitLevel, nextFruitLevel, fruitCount, maxFruitLevel
        const totalSize = staticFeatures + (maxFruits * featuresPerFruit);
        
        const observation = new Float32Array(totalSize);
        
        // Static features
        observation[0] = state.currentFruitLevel;
        observation[1] = state.nextFruitLevel;
        observation[2] = Math.min(state.fruitCount / maxFruits, 1);
        observation[3] = state.maxFruitLevel;
        
        // Fruit features
        let offset = staticFeatures;
        for (let i = 0; i < maxFruits; i++) {
            if (i < state.fruits.length) {
                const fruit = state.fruits[i];
                observation[offset] = fruit.level;
                observation[offset + 1] = fruit.x;
                observation[offset + 2] = fruit.y;
                observation[offset + 3] = fruit.velocityX;
                observation[offset + 4] = fruit.velocityY;
            } else {
                // Pad with zeros for missing fruits
                observation[offset] = 0;
                observation[offset + 1] = 0;
                observation[offset + 2] = 0;
                observation[offset + 3] = 0;
                observation[offset + 4] = 0;
            }
            offset += featuresPerFruit;
        }
        
        return observation;
    }

    /**
     * Computes a reward signal for RL training
     * 
     * @param {Object} [previousState=null] - Previous state for comparison
     * @returns {number} Reward value
     */
    function getReward(previousState = null) {
        const state = getState();
        
        // Game over penalty
        if (state.isGameOver) {
            return -100;
        }
        
        let reward = 0;
        
        // Reward for merges (if previous state provided)
        if (previousState) {
            const scoreDiff = state.score - previousState.score;
            reward += scoreDiff * 0.1; // Scale score difference
            
            // Bonus for reaching new max fruit level
            if (state.maxFruitLevel > previousState.maxFruitLevel) {
                reward += (state.maxFruitLevel - previousState.maxFruitLevel) * 10;
            }
        }
        
        // Small penalty for high fruit stack (approaching game over)
        const fruits = state.fruits;
        if (fruits.length > 0) {
            const minY = Math.min(...fruits.map(f => f.y));
            const fillRatio = 1 - (minY / state.gameHeight);
            if (fillRatio > 0.7) {
                reward -= (fillRatio - 0.7) * 10;
            }
        }
        
        return reward;
    }

    /**
     * Destroys the game instance and cleans up resources
     */
    function destroy() {
        stop();

        if (engine) {
            Events.off(engine);
        }

        if (world) {
            World.clear(world, false);
            Composite.clear(world, false, true);
        }

        if (engine) {
            Engine.clear(engine);
        }

        if (render && render.canvas) {
            render.canvas.remove();
        }

        engine = null;
        render = null;
        runner = null;
        world = null;
        onMerge = null;
        onGameOver = null;
        onScoreUpdate = null;
        onDrop = null;
    }

    // Public API
    return {
        // Lifecycle methods
        init,
        start,
        stop,
        reset,
        destroy,

        // Game actions
        dropFruit,
        step,
        stepMultiple,

        // State getters
        getState,
        getNormalizedState,
        getObservation,
        getReward,

        // Event handlers (setters)
        set onMerge(callback) { onMerge = callback; },
        set onGameOver(callback) { onGameOver = callback; },
        set onScoreUpdate(callback) { onScoreUpdate = callback; },
        set onDrop(callback) { onDrop = callback; },

        // Configuration constants (readonly)
        get GAME_WIDTH() { return gameWorldWidth; },
        get GAME_HEIGHT() { return gameWorldHeight; },
        get GAME_OVER_LINE_Y() { return gameOverLineY; },
        get DROP_AREA_Y() { return dropAreaY; },
        get WALL_THICKNESS() { return wallThickness; },
        get MAX_FRUIT_LEVEL() { return MAX_FRUIT_LEVEL; },
        get FRUITS() { return FRUITS; },
        get STARTING_FRUIT_LEVELS() { return STARTING_FRUIT_LEVELS; },

        // Mode helpers
        get isHeadless() { return headless; },
        get isFastMode() { return fastMode; },

        // Enable/disable fast mode at runtime
        setFastMode(enabled) {
            if (enabled && !fastModeRunning && engine && !isGameOver) {
                startFastMode();
            } else if (!enabled && fastModeRunning) {
                stopFastMode();
                if (runner && engine) {
                    Runner.run(runner, engine);
                }
            }
        }
    };
}

// Export constants for external use
export { FRUITS, MAX_FRUIT_LEVEL, NATIVE_WIDTH, NATIVE_HEIGHT, STARTING_FRUIT_LEVELS };
