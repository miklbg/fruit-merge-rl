/**
 * AI Training Module for Fruit Merge Game
 * 
 * This module provides reinforcement learning training capabilities using TensorFlow.js.
 * It implements a Deep Q-Network (DQN) agent that learns to play the Fruit Merge game.
 * 
 * @module ai-training
 */

import { createGame, NATIVE_WIDTH } from './game-api.js';

/**
 * Creates a DQN (Deep Q-Network) agent for playing Fruit Merge
 * 
 * @param {Object} options - Configuration options
 * @param {Object} options.tf - TensorFlow.js module reference
 * @param {number} [options.observationSize=104] - Size of observation vector
 * @param {number} [options.actionSize=20] - Number of discrete actions (drop positions)
 * @param {number} [options.learningRate=0.001] - Learning rate for optimizer
 * @param {number} [options.gamma=0.99] - Discount factor for future rewards
 * @param {number} [options.epsilon=1.0] - Initial exploration rate
 * @param {number} [options.epsilonMin=0.01] - Minimum exploration rate
 * @param {number} [options.epsilonDecay=0.995] - Exploration decay rate per episode
 * @param {number} [options.batchSize=32] - Batch size for training
 * @param {number} [options.memorySize=10000] - Size of replay memory
 * @returns {Object} DQN agent API
 */
export function createDQNAgent(options = {}) {
    const {
        tf,
        observationSize = 104, // 4 static + 20 fruits * 5 features
        actionSize = 20, // Discretized drop positions
        learningRate = 0.001,
        gamma = 0.99,
        epsilon: initialEpsilon = 1.0,
        epsilonMin = 0.01,
        epsilonDecay = 0.995,
        batchSize = 32,
        memorySize = 10000
    } = options;

    if (!tf) {
        throw new Error('TensorFlow.js module is required. Pass it via options.tf');
    }

    let epsilon = initialEpsilon;
    let model = null;
    let targetModel = null;
    let replayMemory = [];
    let trainStepCount = 0;

    /**
     * Creates the neural network model
     */
    function createModel() {
        const model = tf.sequential();

        // Input layer with observation size
        model.add(tf.layers.dense({
            inputShape: [observationSize],
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Hidden layers
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Output layer - Q-values for each action
        model.add(tf.layers.dense({
            units: actionSize,
            activation: 'linear',
            kernelInitializer: 'heNormal'
        }));

        model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'meanSquaredError'
        });

        return model;
    }

    /**
     * Initializes the agent models
     */
    function init() {
        model = createModel();
        targetModel = createModel();
        updateTargetModel();
    }

    /**
     * Updates the target model with weights from the main model
     */
    function updateTargetModel() {
        const weights = model.getWeights();
        targetModel.setWeights(weights);
    }

    /**
     * Converts action index to game drop position
     * 
     * @param {number} actionIndex - Action index (0 to actionSize-1)
     * @returns {number} X position in game world coordinates
     */
    function actionToPosition(actionIndex) {
        // Map action index to x position across the game width
        const padding = NATIVE_WIDTH * 0.1; // 10% padding on each side
        const usableWidth = NATIVE_WIDTH - 2 * padding;
        return padding + (actionIndex / (actionSize - 1)) * usableWidth;
    }

    /**
     * Selects an action using epsilon-greedy policy
     * 
     * @param {Float32Array} observation - Current observation
     * @returns {number} Selected action index
     */
    function selectAction(observation) {
        if (Math.random() < epsilon) {
            // Exploration: random action
            return Math.floor(Math.random() * actionSize);
        }

        // Exploitation: best action according to model
        return tf.tidy(() => {
            const input = tf.tensor2d([observation], [1, observationSize]);
            const qValues = model.predict(input);
            return qValues.argMax(1).dataSync()[0];
        });
    }

    /**
     * Stores a transition in replay memory
     * 
     * @param {Float32Array} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Float32Array} nextState - Next state
     * @param {boolean} done - Whether episode ended
     */
    function remember(state, action, reward, nextState, done) {
        replayMemory.push({
            state: new Float32Array(state),
            action,
            reward,
            nextState: new Float32Array(nextState),
            done
        });

        // Remove oldest memories if over capacity
        if (replayMemory.length > memorySize) {
            replayMemory.shift();
        }
    }

    /**
     * Samples a batch from replay memory
     * 
     * @returns {Array} Batch of transitions
     */
    function sampleBatch() {
        const batch = [];
        const indices = new Set();

        while (indices.size < Math.min(batchSize, replayMemory.length)) {
            indices.add(Math.floor(Math.random() * replayMemory.length));
        }

        indices.forEach(i => batch.push(replayMemory[i]));
        return batch;
    }

    /**
     * Trains the model on a batch of experiences
     * 
     * @returns {number|null} Training loss or null if not enough samples
     */
    async function train() {
        if (replayMemory.length < batchSize) {
            return null;
        }

        const batch = sampleBatch();
        
        const loss = await tf.tidy(() => {
            // Prepare batch data
            const states = tf.tensor2d(batch.map(t => Array.from(t.state)));
            const nextStates = tf.tensor2d(batch.map(t => Array.from(t.nextState)));
            const actions = batch.map(t => t.action);
            const rewards = batch.map(t => t.reward);
            const dones = batch.map(t => t.done);

            // Get current Q-values
            const currentQs = model.predict(states);
            
            // Get next Q-values from target network
            const nextQs = targetModel.predict(nextStates);
            const maxNextQs = nextQs.max(1).dataSync();

            // Calculate target Q-values
            const targetQsData = currentQs.arraySync();
            for (let i = 0; i < batch.length; i++) {
                if (dones[i]) {
                    targetQsData[i][actions[i]] = rewards[i];
                } else {
                    targetQsData[i][actions[i]] = rewards[i] + gamma * maxNextQs[i];
                }
            }

            const targetQs = tf.tensor2d(targetQsData);

            // Train model
            return model.trainOnBatch(states, targetQs);
        });

        trainStepCount++;

        // Update target network periodically
        if (trainStepCount % 100 === 0) {
            updateTargetModel();
        }

        return loss;
    }

    /**
     * Decays the exploration rate
     */
    function decayEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
    }

    /**
     * Saves the model to browser IndexedDB or downloads
     * 
     * @param {string} [path='indexeddb://fruit-merge-dqn'] - Save path
     */
    async function saveModel(path = 'indexeddb://fruit-merge-dqn') {
        if (model) {
            await model.save(path);
        }
    }

    /**
     * Loads a model from browser IndexedDB or URL
     * 
     * @param {string} [path='indexeddb://fruit-merge-dqn'] - Load path
     */
    async function loadModel(path = 'indexeddb://fruit-merge-dqn') {
        try {
            model = await tf.loadLayersModel(path);
            model.compile({
                optimizer: tf.train.adam(learningRate),
                loss: 'meanSquaredError'
            });
            updateTargetModel();
            return true;
        } catch (error) {
            console.warn('Could not load model:', error);
            return false;
        }
    }

    /**
     * Gets the model summary
     */
    function getSummary() {
        if (model) {
            model.summary();
        }
    }

    /**
     * Disposes of TensorFlow.js tensors and models
     */
    function dispose() {
        if (model) {
            model.dispose();
        }
        if (targetModel) {
            targetModel.dispose();
        }
        model = null;
        targetModel = null;
        replayMemory = [];
    }

    // Public API
    return {
        init,
        selectAction,
        actionToPosition,
        remember,
        train,
        decayEpsilon,
        saveModel,
        loadModel,
        getSummary,
        dispose,
        updateTargetModel,

        // State getters
        get epsilon() { return epsilon; },
        set epsilon(value) { epsilon = value; },
        get memorySize() { return replayMemory.length; },
        get trainStepCount() { return trainStepCount; },
        get actionSize() { return actionSize; }
    };
}

/**
 * Creates a training environment that manages game instances and training loop
 * 
 * @param {Object} options - Configuration options
 * @param {Object} options.Matter - Matter.js module reference
 * @param {Object} options.tf - TensorFlow.js module reference
 * @param {Object} [options.agentOptions={}] - Options for DQN agent
 * @returns {Object} Training environment API
 */
export function createTrainingEnvironment(options = {}) {
    const {
        Matter,
        tf,
        agentOptions = {}
    } = options;

    if (!Matter || !tf) {
        throw new Error('Both Matter.js and TensorFlow.js modules are required');
    }

    let game = null;
    let agent = null;
    let isTraining = false;
    let episodeCount = 0;
    let totalSteps = 0;
    let currentEpisodeReward = 0;
    let currentEpisodeSteps = 0;
    let previousState = null;

    // Training statistics
    const stats = {
        episodeRewards: [],
        episodeLengths: [],
        losses: [],
        epsilonHistory: [],
        avgReward: 0,
        maxScore: 0
    };

    // Callbacks
    let onEpisodeEnd = null;
    let onTrainStep = null;
    let onStatsUpdate = null;

    /**
     * Initializes the training environment
     */
    function init() {
        // Create headless fast game
        game = createGame({
            headless: true,
            fastMode: true,
            Matter
        });

        // Create agent
        agent = createDQNAgent({
            tf,
            ...agentOptions
        });
        agent.init();

        game.init();
    }

    /**
     * Runs a single step in the environment
     * 
     * @returns {Object} Step result
     */
    function step() {
        if (!game || !agent) return null;

        const state = game.getObservation();
        const gameState = game.getState();

        // If game is over or waiting for action
        if (gameState.isGameOver) {
            return { done: true, reward: -100, score: gameState.score };
        }

        // Select and execute action
        const action = agent.selectAction(state);
        const dropX = agent.actionToPosition(action);
        
        const dropped = game.dropFruit(dropX);
        
        if (dropped) {
            // Run physics for a bit to let fruit settle
            game.stepMultiple(30);
            
            const nextState = game.getObservation();
            const reward = game.getReward(previousState ? { score: previousState.score, maxFruitLevel: previousState.maxFruitLevel } : null);
            const nextGameState = game.getState();
            const done = nextGameState.isGameOver;

            // Store experience
            agent.remember(state, action, reward, nextState, done);

            previousState = nextGameState;
            currentEpisodeReward += reward;
            currentEpisodeSteps++;
            totalSteps++;

            return {
                done,
                reward,
                score: nextGameState.score,
                action,
                dropped: true
            };
        } else {
            // Could not drop (still in cooldown), run a few physics steps
            game.stepMultiple(5);
            return { done: false, reward: 0, score: gameState.score, dropped: false };
        }
    }

    /**
     * Runs a complete episode
     * 
     * @param {number} [maxSteps=1000] - Maximum steps per episode
     * @returns {Object} Episode result
     */
    async function runEpisode(maxSteps = 1000) {
        game.reset();
        game.start();

        previousState = null;
        currentEpisodeReward = 0;
        currentEpisodeSteps = 0;

        let stepResult = { done: false };
        let episodeSteps = 0;

        while (!stepResult.done && episodeSteps < maxSteps) {
            stepResult = step();
            episodeSteps++;

            // Train periodically
            if (totalSteps % 4 === 0) {
                const loss = await agent.train();
                if (loss !== null) {
                    stats.losses.push(loss);
                    if (onTrainStep) {
                        onTrainStep({ loss, step: totalSteps });
                    }
                }
            }
        }

        // Record episode stats
        const finalScore = game.getState().score;
        stats.episodeRewards.push(currentEpisodeReward);
        stats.episodeLengths.push(currentEpisodeSteps);
        stats.epsilonHistory.push(agent.epsilon);

        if (finalScore > stats.maxScore) {
            stats.maxScore = finalScore;
        }

        // Calculate running average
        const recentRewards = stats.episodeRewards.slice(-100);
        stats.avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;

        episodeCount++;
        agent.decayEpsilon();

        const episodeResult = {
            episode: episodeCount,
            reward: currentEpisodeReward,
            score: finalScore,
            steps: currentEpisodeSteps,
            epsilon: agent.epsilon,
            avgReward: stats.avgReward,
            maxScore: stats.maxScore
        };

        if (onEpisodeEnd) {
            onEpisodeEnd(episodeResult);
        }

        if (onStatsUpdate) {
            onStatsUpdate(stats);
        }

        return episodeResult;
    }

    /**
     * Runs multiple training episodes
     * 
     * @param {number} numEpisodes - Number of episodes to run
     * @param {number} [maxStepsPerEpisode=1000] - Maximum steps per episode
     * @param {Function} [progressCallback=null] - Called after each episode
     */
    async function train(numEpisodes, maxStepsPerEpisode = 1000, progressCallback = null) {
        isTraining = true;

        for (let i = 0; i < numEpisodes && isTraining; i++) {
            const result = await runEpisode(maxStepsPerEpisode);

            if (progressCallback) {
                progressCallback(result, i + 1, numEpisodes);
            }

            // Save model periodically
            if ((i + 1) % 100 === 0) {
                await agent.saveModel();
            }

            // Yield to browser to prevent freezing
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        isTraining = false;
    }

    /**
     * Stops the training loop
     */
    function stopTraining() {
        isTraining = false;
    }

    /**
     * Evaluates the agent without training
     * 
     * @param {number} [numEpisodes=10] - Number of evaluation episodes
     * @returns {Object} Evaluation results
     */
    async function evaluate(numEpisodes = 10) {
        const originalEpsilon = agent.epsilon;
        agent.epsilon = 0; // No exploration during evaluation

        const scores = [];
        const rewards = [];

        for (let i = 0; i < numEpisodes; i++) {
            game.reset();
            game.start();

            previousState = null;
            let episodeReward = 0;
            let stepResult = { done: false };
            let steps = 0;

            while (!stepResult.done && steps < 1000) {
                stepResult = step();
                episodeReward += stepResult.reward || 0;
                steps++;
            }

            scores.push(game.getState().score);
            rewards.push(episodeReward);

            await new Promise(resolve => setTimeout(resolve, 0));
        }

        agent.epsilon = originalEpsilon;

        return {
            avgScore: scores.reduce((a, b) => a + b, 0) / scores.length,
            maxScore: Math.max(...scores),
            minScore: Math.min(...scores),
            avgReward: rewards.reduce((a, b) => a + b, 0) / rewards.length,
            scores,
            rewards
        };
    }

    /**
     * Gets training statistics
     */
    function getStats() {
        return { ...stats };
    }

    /**
     * Saves the current model
     */
    async function saveModel(path) {
        if (agent) {
            await agent.saveModel(path);
        }
    }

    /**
     * Loads a saved model
     */
    async function loadModel(path) {
        if (agent) {
            return await agent.loadModel(path);
        }
        return false;
    }

    /**
     * Cleans up resources
     */
    function dispose() {
        isTraining = false;

        if (game) {
            game.destroy();
            game = null;
        }

        if (agent) {
            agent.dispose();
            agent = null;
        }
    }

    // Public API
    return {
        init,
        step,
        runEpisode,
        train,
        stopTraining,
        evaluate,
        getStats,
        saveModel,
        loadModel,
        dispose,

        // Getters
        get isTraining() { return isTraining; },
        get episodeCount() { return episodeCount; },
        get totalSteps() { return totalSteps; },
        get agent() { return agent; },
        get game() { return game; },

        // Event handlers
        set onEpisodeEnd(callback) { onEpisodeEnd = callback; },
        set onTrainStep(callback) { onTrainStep = callback; },
        set onStatsUpdate(callback) { onStatsUpdate = callback; }
    };
}
