/**
 * Reinforcement Learning Interface for Fruit Merge Game
 * 
 * This module provides a generic RL API that wraps the existing game logic.
 * It only reads game data and exposes a clean interface for RL agents.
 * 
 * The interface follows the OpenAI Gym-style API pattern:
 * - getObservation(): Returns the current game state
 * - step(action): Executes an action and returns observation, reward, done, info
 * - reset(): Resets the environment and returns initial observation
 * - getActionSpace(): Returns the action space definition
 * - getObservationSpace(): Returns the observation space definition
 * 
 * @module rl-interface
 */

/**
 * Create an RL environment wrapper for the Fruit Merge game
 * 
 * @param {Object} gameState - Reference to the game state object
 * @param {Object} config - Configuration options
 * @param {number} config.numDropPositions - Number of discrete drop positions (default: 20)
 * @param {number} config.rewardScale - Scale factor for rewards (default: 1.0)
 * @returns {Object} RL environment interface
 */
export function createRLEnvironment(gameState, config = {}) {
    const {
        numDropPositions = 20,
        rewardScale = 1.0
    } = config;

    // Track state for reward calculation
    let previousScore = 0;
    let stepCount = 0;

    /**
     * Get the observation space definition
     * Describes the structure and bounds of observations
     * 
     * @returns {Object} Observation space specification
     */
    function getObservationSpace() {
        return {
            // Current fruit level to drop (0-3 for starting fruits)
            currentFruitLevel: { type: 'discrete', min: 0, max: 9, description: 'Current fruit level to drop' },
            
            // Next fruit level (0-3 for starting fruits)
            nextFruitLevel: { type: 'discrete', min: 0, max: 9, description: 'Next fruit level in queue' },
            
            // Current score
            score: { type: 'continuous', min: 0, max: Infinity, description: 'Current game score' },
            
            // Fruits in the game (variable length array of fruit objects)
            fruits: {
                type: 'array',
                maxLength: 50,
                itemSchema: {
                    x: { type: 'continuous', min: 0, max: 1, description: 'Normalized x position (0-1)' },
                    y: { type: 'continuous', min: 0, max: 1, description: 'Normalized y position (0-1)' },
                    level: { type: 'discrete', min: 0, max: 9, description: 'Fruit level' },
                    velocityX: { type: 'continuous', min: -50, max: 50, description: 'X velocity' },
                    velocityY: { type: 'continuous', min: -50, max: 50, description: 'Y velocity' },
                    radius: { type: 'continuous', min: 0, max: 1, description: 'Normalized radius' }
                },
                description: 'Array of fruits currently in play'
            },
            
            // Game status flags
            isGameOver: { type: 'boolean', description: 'Whether the game has ended' },
            canDrop: { type: 'boolean', description: 'Whether a fruit can be dropped now' },
            
            // Warning status
            isWarningActive: { type: 'boolean', description: 'Whether fruits are near the game over line' }
        };
    }

    /**
     * Get the action space definition
     * 
     * @returns {Object} Action space specification
     */
    function getActionSpace() {
        return {
            type: 'discrete',
            n: numDropPositions,
            description: `Discrete drop position from 0 to ${numDropPositions - 1} across the game width`
        };
    }

    /**
     * Convert a discrete action to game world x coordinate
     * 
     * @param {number} action - Discrete action (0 to numDropPositions-1)
     * @returns {number} Game world x coordinate
     */
    function actionToPosition(action) {
        const {
            gameWorldWidth,
            wallThickness,
            currentFruitLevel,
            FRUITS
        } = gameState;

        // Get current fruit radius for boundary calculation
        const fruitData = FRUITS[currentFruitLevel];
        const radius = fruitData ? fruitData.baseRadius : 0;

        // Calculate playable width (excluding walls and fruit radius)
        const minX = radius + wallThickness;
        const maxX = gameWorldWidth - radius - wallThickness;
        const playableWidth = maxX - minX;

        // Map action to position within playable area
        const normalizedAction = action / (numDropPositions - 1);
        return minX + normalizedAction * playableWidth;
    }

    /**
     * Get the current observation (game state)
     * 
     * @returns {Object} Current observation
     */
    function getObservation() {
        const {
            currentFruitLevel,
            nextFruitLevel,
            score,
            isGameOver,
            isDropping,
            isWarningActive,
            gameWorldWidth,
            gameWorldHeight,
            world,
            Matter,
            FRUITS,
            lastDropTime,
            DROP_COOLDOWN_MS
        } = gameState;

        // Extract fruit positions from Matter.js world
        const fruits = [];
        if (world && Matter) {
            const bodies = Matter.Composite.allBodies(world);
            for (const body of bodies) {
                if (body.label === 'fruit') {
                    const fruitData = FRUITS[body.fruitLevel];
                    fruits.push({
                        x: body.position.x / gameWorldWidth,
                        y: body.position.y / gameWorldHeight,
                        level: body.fruitLevel,
                        velocityX: body.velocity.x,
                        velocityY: body.velocity.y,
                        radius: (fruitData ? fruitData.baseRadius : 0) / gameWorldWidth
                    });
                }
            }
        }

        // Determine if we can drop (not currently dropping and cooldown passed)
        const currentTime = Date.now();
        const canDrop = !isDropping && 
                        !isGameOver && 
                        (currentTime - lastDropTime >= DROP_COOLDOWN_MS);

        return {
            currentFruitLevel: currentFruitLevel || 0,
            nextFruitLevel: nextFruitLevel || 0,
            score: score || 0,
            fruits,
            isGameOver: isGameOver || false,
            canDrop,
            isWarningActive: isWarningActive || false
        };
    }

    /**
     * Execute an action and return the result
     * 
     * @param {number} action - The action to take (drop position index)
     * @returns {Object} Result containing observation, reward, done, and info
     */
    function step(action) {
        // Validate action
        if (action < 0 || action >= numDropPositions) {
            throw new Error(`Invalid action: ${action}. Must be between 0 and ${numDropPositions - 1}`);
        }

        const observation = getObservation();
        
        // If we can't drop, return current state with no reward
        if (!observation.canDrop) {
            return {
                observation,
                reward: 0,
                done: observation.isGameOver,
                info: { 
                    actionExecuted: false, 
                    reason: observation.isGameOver ? 'game_over' : 'cooldown_active' 
                }
            };
        }

        // Get the callbacks if they exist
        const { moveFruit, dropFruit } = gameState;

        // Convert action to position and execute
        const position = actionToPosition(action);
        
        // Move the preview fruit to the position (in game world coordinates)
        if (typeof moveFruit === 'function') {
            moveFruit(position, true); // true indicates internal call (already in world coords)
        }

        // Drop the fruit
        if (typeof dropFruit === 'function') {
            dropFruit();
        }

        stepCount++;

        // Wait a brief moment for physics to update (in real usage, 
        // the caller should manage this timing)
        const newObservation = getObservation();

        // Calculate reward
        const reward = calculateReward(newObservation, observation);
        previousScore = newObservation.score;

        return {
            observation: newObservation,
            reward,
            done: newObservation.isGameOver,
            info: {
                actionExecuted: true,
                dropPosition: position,
                stepCount,
                scoreDelta: newObservation.score - observation.score
            }
        };
    }

    /**
     * Calculate the reward for a state transition
     * 
     * @param {Object} newObs - New observation after action
     * @param {Object} oldObs - Observation before action
     * @returns {number} Calculated reward
     */
    function calculateReward(newObs, oldObs) {
        let reward = 0;

        // Reward for score increase (merging fruits)
        const scoreDelta = newObs.score - oldObs.score;
        if (scoreDelta > 0) {
            reward += scoreDelta * rewardScale;
        }

        // Penalty for game over
        if (newObs.isGameOver && !oldObs.isGameOver) {
            reward -= 100 * rewardScale;
        }

        // Small penalty for being in warning zone (fruits too high)
        if (newObs.isWarningActive) {
            reward -= 1 * rewardScale;
        }

        return reward;
    }

    /**
     * Reset the environment (requires game reset callback)
     * 
     * @returns {Object} Initial observation after reset
     */
    function reset() {
        const { handleRestart } = gameState;
        
        if (typeof handleRestart === 'function') {
            handleRestart();
        }

        previousScore = 0;
        stepCount = 0;

        return getObservation();
    }

    /**
     * Get environment metadata
     * 
     * @returns {Object} Environment metadata
     */
    function getInfo() {
        return {
            name: 'FruitMerge-v1',
            description: 'Fruit Merge puzzle game RL environment',
            actionSpace: getActionSpace(),
            observationSpace: getObservationSpace(),
            rewardRange: [-100 * rewardScale, Infinity],
            config: {
                numDropPositions,
                rewardScale
            }
        };
    }

    // Return the RL environment interface
    return {
        getObservation,
        getObservationSpace,
        getActionSpace,
        step,
        reset,
        getInfo,
        actionToPosition
    };
}

/**
 * Helper class for managing game state references
 * Use this to collect references from the game's scope
 */
export class GameStateCollector {
    constructor() {
        this.state = {};
    }

    /**
     * Register a reference with the collector
     * 
     * @param {string} key - The key name for this reference
     * @param {*} value - The value or reference to store
     */
    register(key, value) {
        this.state[key] = value;
    }

    /**
     * Register multiple references at once
     * 
     * @param {Object} refs - Object containing key-value pairs to register
     */
    registerMany(refs) {
        Object.assign(this.state, refs);
    }

    /**
     * Get all registered state
     * 
     * @returns {Object} All registered state references
     */
    getState() {
        return this.state;
    }

    /**
     * Create an RL environment using the collected state
     * 
     * @param {Object} config - Configuration options for the RL environment
     * @returns {Object} RL environment interface
     */
    createEnvironment(config = {}) {
        return createRLEnvironment(this.state, config);
    }
}
