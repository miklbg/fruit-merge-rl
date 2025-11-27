# Reinforcement Learning Interface

This module provides a generic RL (Reinforcement Learning) API for the Fruit Merge game, following the OpenAI Gym-style interface pattern.

## Overview

The RL interface wraps the existing game logic without modifying it, providing a clean API for RL agents to:
- Observe the game state
- Take actions (drop fruits at specific positions)
- Receive rewards based on game events
- Reset the environment

## Installation

The RL interface is included in the game's JavaScript modules. Import it in your code:

```javascript
import { createRLEnvironment, GameStateCollector } from './rl/rl-interface.js';
```

## Quick Start

### Method 1: Using GameStateCollector (Recommended)

```javascript
import { GameStateCollector } from './rl/rl-interface.js';

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
    Matter,
    
    // Game control functions
    moveFruit,
    dropFruit,
    handleRestart
});

// Create the RL environment
const env = collector.createEnvironment({
    numDropPositions: 20,  // Number of discrete actions
    rewardScale: 1.0       // Scale factor for rewards
});
```

### Method 2: Direct Creation

```javascript
import { createRLEnvironment } from './rl/rl-interface.js';

const gameState = {
    // ... your game state references
};

const env = createRLEnvironment(gameState, {
    numDropPositions: 20,
    rewardScale: 1.0
});
```

## API Reference

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
//     description: 'Fruit Merge puzzle game RL environment',
//     actionSpace: {...},
//     observationSpace: {...},
//     rewardRange: [-100, Infinity],
//     config: { numDropPositions: 20, rewardScale: 1.0 }
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

## Example: Simple Random Agent

```javascript
import { GameStateCollector } from './rl/rl-interface.js';

// After game initialization, create the environment
const collector = new GameStateCollector();
// ... register game state ...
const env = collector.createEnvironment();

// Simple game loop
async function runEpisode() {
    let obs = env.reset();
    let totalReward = 0;
    
    while (!obs.isGameOver) {
        // Wait for drop cooldown
        if (obs.canDrop) {
            // Random action
            const action = Math.floor(Math.random() * 20);
            const result = env.step(action);
            
            totalReward += result.reward;
            obs = result.observation;
            
            console.log(`Step reward: ${result.reward}, Total: ${totalReward}`);
        }
        
        // Delay to let physics update and cooldown to pass
        // Use at least 500ms to account for DROP_COOLDOWN_MS (400ms) plus physics simulation
        await new Promise(r => setTimeout(r, 500));
        obs = env.getObservation();
    }
    
    console.log(`Episode finished with score: ${obs.score}`);
}
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

- The RL interface only **reads** game state and calls existing game functions
- No core gameplay logic is modified
- Actions during cooldown periods are ignored (check `canDrop` before stepping)
- The `fruits` array may be empty if no fruits are in play yet
- Rewards are calculated immediately after each step, but merges happen over time due to physics simulation
