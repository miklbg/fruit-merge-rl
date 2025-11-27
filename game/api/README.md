# Game API

This module provides a generic API for the Fruit Merge game, designed for programmatic control including reinforcement learning training with TensorFlow.js or other frameworks.

## Overview

The Game API wraps the existing game logic without modifying it, providing a clean interface for:
- Observing the game state
- Taking actions (drop fruits at specific positions)
- Receiving rewards based on game events
- Resetting the environment
- **Training mode**: Disabling rendering/UI for faster RL training

## Installation

The Game API is included in the game's JavaScript modules. Import it in your code:

```javascript
import { createGameAPI, GameStateCollector } from './api/game-api.js';
```

## Quick Start

### Method 1: Using GameStateCollector (Recommended)

```javascript
import { GameStateCollector } from './api/game-api.js';

// Create a collector and register game state references
const collector = new GameStateCollector();

collector.registerMany({
    // Game state variables
    currentFruitLevel,
    nextFruitLevel,
    score,
    isGameOver,
    isDropping,
    isWarningActive,
    lastDropTime,
    
    // Game constants
    gameWorldWidth: NATIVE_WIDTH,
    gameWorldHeight: NATIVE_HEIGHT,
    wallThickness,
    FRUITS,
    DROP_COOLDOWN_MS,
    
    // Matter.js references
    world,
    engine,  // Required for training mode
    render,  // Required for training mode
    Matter,
    
    // Game control functions
    moveFruit,
    dropFruit,
    handleRestart
});

// Create the Game API
const env = collector.createEnvironment({
    numDropPositions: 20,  // Number of discrete actions
    rewardScale: 1.0       // Scale factor for rewards
});
```

### Method 2: Direct Creation

```javascript
import { createGameAPI } from './api/game-api.js';

const gameState = {
    // ... your game state references
};

const env = createGameAPI(gameState, {
    numDropPositions: 20,
    rewardScale: 1.0
});
```

## Training Mode

Training mode disables rendering and UI updates for faster RL training with frameworks like TensorFlow.js.

### Enabling Training Mode

```javascript
// Disable rendering for fast training
env.enableTrainingMode();

// Run training loop
for (let episode = 0; episode < 1000; episode++) {
    let obs = env.reset();
    
    while (!obs.isGameOver) {
        const action = agent.selectAction(obs);  // Your RL agent
        const result = env.step(action);
        
        // Manually step physics (faster than waiting for requestAnimationFrame)
        // 30 steps * ~16.67ms = ~500ms of simulated physics time
        env.runPhysicsSteps(30);
        
        agent.learn(result);  // Train your agent
        obs = result.observation;
    }
}

// Re-enable rendering when done
env.disableTrainingMode();
```

### Training Mode API

#### `enableTrainingMode()`

Disables rendering and UI updates for faster training.

```javascript
const success = env.enableTrainingMode();
// Returns: true if training mode was enabled successfully
```

#### `disableTrainingMode()`

Re-enables rendering and UI updates.

```javascript
const success = env.disableTrainingMode();
// Returns: true if training mode was disabled successfully
```

#### `stepPhysics(deltaTime)`

Manually step the physics engine forward.

```javascript
env.stepPhysics(16.67);  // One frame at 60fps
```

#### `runPhysicsSteps(steps, deltaTime)`

Run multiple physics steps at once. The default deltaTime is ~16.67ms (1000/60) for 60fps physics.

```javascript
// Run 30 physics steps (~500ms of simulated time at 60fps)
const obs = env.runPhysicsSteps(30);
// Returns: observation after all steps complete

// Or specify custom deltaTime
const obs2 = env.runPhysicsSteps(30, 1000/60);
```

#### `isInTrainingMode()`

Check if currently in training mode.

```javascript
if (env.isInTrainingMode()) {
    console.log('Training mode is active');
}
```

## Core API Reference

### Environment Methods

#### `getObservation()`

Returns the current game state as an observation object.

```javascript
const obs = env.getObservation();
// Returns:
// {
//     currentFruitLevel: 0,      // Current fruit to drop (0-9)
//     nextFruitLevel: 1,         // Next fruit in queue (0-9)
//     score: 150,                // Current score
//     fruits: [...],             // Array of fruit objects in play
//     isGameOver: false,         // Game over flag
//     canDrop: true,             // Can drop a fruit now?
//     isWarningActive: false     // Fruits near game over line?
// }
```

#### `step(action)`

Execute an action and return the result.

```javascript
const result = env.step(10);  // Drop at position 10 (middle)
// Returns:
// {
//     observation: {...},        // New observation after action
//     reward: 5,                 // Reward for this step
//     done: false,               // Episode finished?
//     info: {                    // Additional info
//         actionExecuted: true,
//         dropPosition: 600,     // Actual x coordinate
//         stepCount: 1,
//         scoreDelta: 5
//     }
// }
```

#### `reset()`

Reset the environment to initial state.

```javascript
const initialObs = env.reset();
```

#### `getActionSpace()`

Get the action space definition.

```javascript
const actionSpace = env.getActionSpace();
// Returns:
// {
//     type: 'discrete',
//     n: 20,
//     description: 'Discrete drop position from 0 to 19 across the game width'
// }
```

#### `getObservationSpace()`

Get the observation space definition.

```javascript
const obsSpace = env.getObservationSpace();
// Returns detailed schema of the observation structure
```

#### `getInfo()`

Get environment metadata.

```javascript
const info = env.getInfo();
// Returns:
// {
//     name: 'FruitMerge-v1',
//     description: 'Fruit Merge puzzle game environment',
//     actionSpace: {...},
//     observationSpace: {...},
//     rewardRange: [-100, Infinity],
//     config: { numDropPositions: 20, rewardScale: 1.0 },
//     isTrainingMode: false
// }
```

#### `actionToPosition(action)`

Convert a discrete action to game world x coordinate.

```javascript
const xPos = env.actionToPosition(10);  // Returns x coordinate for action 10
```

## Observation Space

The observation includes:

| Field | Type | Description |
|-------|------|-------------|
| `currentFruitLevel` | int (0-9) | Level of the fruit about to be dropped |
| `nextFruitLevel` | int (0-9) | Level of the next fruit in queue |
| `score` | int | Current game score |
| `fruits` | array | Array of fruit objects currently in play |
| `isGameOver` | bool | Whether the game has ended |
| `canDrop` | bool | Whether a drop action can be executed now |
| `isWarningActive` | bool | Whether fruits are dangerously high |

### Fruit Object Schema

Each fruit in the `fruits` array contains:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `x` | float | 0-1 | Normalized x position |
| `y` | float | 0-1 | Normalized y position |
| `level` | int | 0-9 | Fruit level |
| `velocityX` | float | -50 to 50 | X velocity |
| `velocityY` | float | -50 to 50 | Y velocity |
| `radius` | float | 0-1 | Normalized radius |

## Action Space

- **Type**: Discrete
- **Size**: Configurable (default: 20 positions)
- **Action 0**: Left edge of playable area
- **Action N-1**: Right edge of playable area

## Reward Structure

The default reward function provides:

| Event | Reward |
|-------|--------|
| Score increase | `+scoreDelta * rewardScale` |
| Game over | `-100 * rewardScale` |
| Warning active | `-1 * rewardScale` per step |

## Example: TensorFlow.js Training Loop

```javascript
import * as tf from '@tensorflow/tfjs';
import { GameStateCollector } from './api/game-api.js';

// Setup environment
const collector = new GameStateCollector();
// ... register game state ...
const env = collector.createEnvironment({ numDropPositions: 20 });

// Enable training mode for faster iteration
env.enableTrainingMode();

// Simple DQN training loop
async function train(agent, episodes = 1000) {
    for (let ep = 0; ep < episodes; ep++) {
        let obs = env.reset();
        let totalReward = 0;
        
        while (!obs.isGameOver) {
            if (obs.canDrop) {
                // Get action from agent
                const stateTensor = tf.tensor2d([observationToArray(obs)]);
                const action = agent.selectAction(stateTensor);
                stateTensor.dispose();
                
                // Execute action
                const result = env.step(action);
                
                // Run physics simulation
                env.runPhysicsSteps(30);
                
                // Store transition and train
                agent.remember(obs, action, result.reward, result.observation, result.done);
                await agent.train();
                
                totalReward += result.reward;
                obs = result.observation;
            } else {
                // Wait for physics to settle
                env.runPhysicsSteps(5);
                obs = env.getObservation();
            }
        }
        
        console.log(`Episode ${ep}: Score ${obs.score}, Total Reward ${totalReward}`);
    }
}

// Re-enable rendering when done
env.disableTrainingMode();
```

## Fruit Levels Reference

| Level | Fruit | Points |
|-------|-------|--------|
| 0 | Blueberry | 5 |
| 1 | Strawberry | 10 |
| 2 | Grapes | 20 |
| 3 | Orange | 35 |
| 4 | Apple | 55 |
| 5 | Lemon | 80 |
| 6 | Cantaloupe | 110 |
| 7 | Pineapple | 150 |
| 8 | Coconut | 200 |
| 9 | Watermelon | 300 |

## Notes

- The Game API only **reads** game state and calls existing game functions
- No core gameplay logic is modified
- Actions during cooldown periods are ignored (check `canDrop` before stepping)
- The `fruits` array may be empty if no fruits are in play yet
- Rewards are calculated immediately after each step, but merges happen over time due to physics simulation
- In training mode, use `stepPhysics()` or `runPhysicsSteps()` to advance the simulation
- For TensorFlow.js, consider using `tf.tidy()` to prevent memory leaks during training
