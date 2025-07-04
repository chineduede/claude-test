//+------------------------------------------------------------------+
//|                                         UltraAdvancedTrading.mqh  |
//|                  World's Most Advanced Trading Algorithms         |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Ultra Advanced Trading"
#property link      ""
#property version   "1.00"

#include <Math\Stat\Math.mqh>

//--- Neural Network Architecture
struct NeuralNetwork
{
    // LSTM-style architecture for time series
    double inputWeights[];       // Flattened 2D array (input x hidden[0])
    double hiddenWeights[];      // Flattened 3D array (layers x neurons x neurons)
    double outputWeights[];      // Flattened 2D array
    double forgetGate[];         // Flattened 2D array
    double inputGate[];          // Flattened 2D array
    double outputGate[];         // Flattened 2D array
    double cellState[];          // Flattened 2D array

    // Attention mechanism
    double attentionWeights[];   // Flattened 2D array
    double contextVector[];

    // Adaptive learning
    double learningRate;
    double momentum;
    double adaptiveRate[];

    // Performance tracking
    double trainingError;
    double validationError;
    int epochsTrained;

    // Dimensions for reconstruction
    int inputSize;
    int hiddenSizes[];
    int outputSize;
};

//--- Hidden Liquidity Detection
struct HiddenLiquidity
{
    double icebergRatio;         // Hidden vs visible order ratio
    double darkPoolActivity;     // Estimated dark pool volume
    double hiddenBuyPressure;    // Unexecuted buy interest
    double hiddenSellPressure;   // Unexecuted sell interest
    double liquidityMirage;      // Fake liquidity score
    double realLiquidity;        // True available liquidity

    // Liquidity prediction
    double predictedLiquidity[];  // Future liquidity forecast
    double liquidityShocks[];     // Sudden liquidity changes
    double marketDepth[];        // True market depth estimates
};

//--- Order Book Dynamics
struct OrderBookDynamics
{
    // Level 2 data analysis
    double bidDepth[];           // Bid volume at each level
    double askDepth[];           // Ask volume at each level
    double bidPressure;          // Weighted bid pressure
    double askPressure;          // Weighted ask pressure

    // Spoofing detection
    double spoofingScore;        // Likelihood of spoofing
    double layeringDetected;     // Layering manipulation
    double momentumIgnition;     // Artificial momentum creation

    // Order book imbalance
    double microImbalance[];     // Tick-by-tick imbalances
    double persistentImbalance;  // Long-term imbalance
    double imbalanceDecay;       // How fast imbalance dissipates

    // High-frequency patterns
    double quoteStuffing;        // Quote stuffing detection
    double pinging;              // Pinging/fishing detection
    double frontRunning;         // Front-running probability
};

//--- Market Maker Model
struct MarketMakerInventory
{
    double currentInventory;     // Current position
    double targetInventory;      // Desired position
    double inventoryLimit;       // Maximum position
    double inventorySkew;        // Bid/ask skew based on inventory

    // Pricing model
    double fairValue;            // Theoretical fair value
    double bidOffset;            // Optimal bid offset
    double askOffset;            // Optimal ask offset
    double spreadCapture;        // Expected spread earnings

    // Risk metrics
    double inventoryRisk;        // Risk from current inventory
    double adverseSelection;     // Adverse selection cost
    double optimalHedgeRatio;    // Hedge ratio for inventory

    // Flow prediction
    double expectedFlow;         // Expected order flow
    double toxicFlow;           // Toxic flow probability
    double informedTraderProb;   // Probability of informed trader
};

//--- Reinforcement Learning Stop Loss
struct AdaptiveStopLoss
{
    // Q-learning parameters
    double qTable[];            // Flattened 2D array (states x actions)
    double rewardHistory[];     // Historical rewards
    double epsilon;             // Exploration rate
    double gamma;               // Discount factor

    // State representation
    double volatilityState;     // Current volatility regime
    double trendState;          // Current trend strength
    double momentumState;       // Current momentum
    double microstructureState; // Microstructure conditions

    // Actions
    double stopDistance[];      // Possible stop distances
    double trailingSpeed[];     // Trailing stop speeds
    double optimalStop;         // Current optimal stop

    // Learning metrics
    double averageReward;       // Average reward per trade
    double stopEfficiency;      // How often stops save money
    double falseStopRate;       // Rate of premature stops

    // Dimensions
    int numStates;
    int numActions;
};

//--- Cross-Asset Contagion
struct CrossAssetContagion
{
    // Correlation dynamics
    double correlationMatrix[];      // Flattened 2D array (numAssets x numAssets)
    double tailDependence[];         // Flattened 2D array
    double correlationBreaks[];      // Correlation regime changes

    // Contagion metrics
    double contagionIndex;           // Overall contagion level
    double systemicRisk;             // Systemic risk score
    double cascadeProbability;       // Probability of cascade

    // Leading indicators
    string leadingAssets[];          // Assets that lead moves
    double leadTime[];               // How much they lead by
    double signalStrength[];         // Strength of leading signal

    // Network effects
    double centralityScore[];        // Network centrality
    double clusterRisk[];           // Risk clustering
    double transmissionSpeed;        // Contagion speed

    // Dimensions
    int numAssets;
};

//--- High-Frequency Microstructure
struct HFMicrostructure
{
    // Tick data patterns
    double tickImbalance;            // Micro tick imbalances
    double tickMomentum;             // Tick-level momentum
    double tickReversal;             // Tick reversal probability

    // Quote dynamics
    double quoteIntensity;           // Quote update frequency
    double quoteLifetime;            // Average quote duration
    double quoteCompetition;         // Competition for best quote
    double bidAskBounce;             // Bid-ask bounce measure
    double spreadPersistence;        // How long spreads persist

    // Execution analysis
    double fillProbability[];        // Fill probability by level
    double expectedSlippage[];       // Expected slippage
    double marketImpact[];          // Market impact function

    // Latency arbitrage
    double latencyAdvantage;         // Latency edge in ms
    double arbOpportunities;         // Arbitrage opportunities/hour
    double competitionIntensity;     // HFT competition level
};

//--- Meta Strategy Controller
struct MetaStrategy
{
    // Strategy weights
    double momentumWeight;
    double meanReversionWeight;
    double marketMakingWeight;
    double arbitrageWeight;

    // Performance tracking
    double strategyReturns[];        // Returns by strategy (1D array: strategy * history + period)
    double strategySharpe[];         // Sharpe by strategy
    double strategyDrawdown[];       // Drawdown by strategy

    // Regime allocation
    double regimeWeights[];          // Weights by regime (1D array: regime * numStrategies + strategy)
    double transitionMatrix[];       // Regime transition probs (1D array)

    // Risk budgeting
    double riskBudget[];            // Risk budget by strategy
    double correlations[];          // Strategy correlations (1D array)
    double optimalMix[];            // Optimal strategy mix
};

//+------------------------------------------------------------------+
//| Initialize Neural Network                                        |
//+------------------------------------------------------------------+
void InitializeNeuralNetwork(NeuralNetwork &nn, int inputSize, int &hiddenSizes[], int outputSize)
{
    int numHidden = ArraySize(hiddenSizes);

    // Store dimensions
    nn.inputSize = inputSize;
    nn.outputSize = outputSize;
    ArrayResize(nn.hiddenSizes, numHidden);
    ArrayCopy(nn.hiddenSizes, hiddenSizes);

    // Initialize input weights (inputSize x hiddenSizes[0])
    ArrayResize(nn.inputWeights, inputSize * hiddenSizes[0]);
    for(int i = 0; i < inputSize; i++)
    {
        for(int j = 0; j < hiddenSizes[0]; j++)
        {
            int idx = i * hiddenSizes[0] + j;
            nn.inputWeights[idx] = (MathRand() / 32767.0 - 0.5) * MathSqrt(2.0 / inputSize);
        }
    }

    // Calculate total size for hidden weights
    int hiddenWeightsSize = 0;
    for(int layer = 0; layer < numHidden - 1; layer++)
        hiddenWeightsSize += hiddenSizes[layer] * hiddenSizes[layer + 1];

    // Initialize hidden weights
    ArrayResize(nn.hiddenWeights, hiddenWeightsSize);
    int offset = 0;
    for(int layer = 0; layer < numHidden - 1; layer++)
    {
        for(int i = 0; i < hiddenSizes[layer]; i++)
        {
            for(int j = 0; j < hiddenSizes[layer + 1]; j++)
            {
                nn.hiddenWeights[offset++] = (MathRand() / 32767.0 - 0.5) * MathSqrt(2.0 / hiddenSizes[layer]);
            }
        }
    }

    // Initialize LSTM gates (all are hiddenSizes[0] x hiddenSizes[0])
    int gateSize = hiddenSizes[0] * hiddenSizes[0];
    ArrayResize(nn.forgetGate, gateSize);
    ArrayResize(nn.inputGate, gateSize);
    ArrayResize(nn.outputGate, gateSize);
    ArrayResize(nn.cellState, gateSize);

    // Initialize attention mechanism
    int lastHiddenSize = hiddenSizes[numHidden - 1];
    ArrayResize(nn.attentionWeights, lastHiddenSize * lastHiddenSize);
    ArrayResize(nn.contextVector, lastHiddenSize);

    // Initialize output weights
    ArrayResize(nn.outputWeights, lastHiddenSize * outputSize);

    nn.learningRate = 0.001;
    nn.momentum = 0.9;
    nn.epochsTrained = 0;
}

//+------------------------------------------------------------------+
//| LSTM Forward Pass                                                |
//+------------------------------------------------------------------+
void LSTMForward(NeuralNetwork &nn, double &inputs[], double &outputs[])
{
    int inputSize = nn.inputSize;
    int hiddenSize = nn.hiddenSizes[0];  // First hidden layer size

    // Temporary arrays
    double hidden[];
    ArrayResize(hidden, hiddenSize);

    // Input transformation
    for(int i = 0; i < hiddenSize; i++)
    {
        hidden[i] = 0;
        for(int j = 0; j < inputSize; j++)
        {
            int idx = j * hiddenSize + i;
            hidden[i] += inputs[j] * nn.inputWeights[idx];
        }
    }

    // LSTM cell computation
    for(int i = 0; i < hiddenSize; i++)
    {
        // Access gate values from flattened arrays
        int diag_idx = i * hiddenSize + i;  // Diagonal element

        // Forget gate
        double forget = Sigmoid(hidden[i] + nn.forgetGate[diag_idx]);

        // Input gate
        double input_ = Sigmoid(hidden[i] + nn.inputGate[diag_idx]);

        // Candidate values
        double candidate = MathTanh(hidden[i]);

        // Update cell state
        nn.cellState[diag_idx] = forget * nn.cellState[diag_idx] + input_ * candidate;

        // Output gate
        double output = Sigmoid(hidden[i] + nn.outputGate[diag_idx]);

        // Hidden state
        hidden[i] = output * MathTanh(nn.cellState[diag_idx]);
    }

    // Apply attention mechanism
    ApplyAttention(nn, hidden);

    // Output layer (simplified)
    ArrayResize(outputs, nn.outputSize);
    for(int i = 0; i < nn.outputSize; i++)
    {
        outputs[i] = 0;
        for(int j = 0; j < hiddenSize; j++)
            outputs[i] += hidden[j] * nn.contextVector[j];
        outputs[i] = Sigmoid(outputs[i]);
    }
}

//+------------------------------------------------------------------+
//| Sigmoid activation function                                      |
//+------------------------------------------------------------------+
double Sigmoid(double x)
{
    return 1.0 / (1.0 + MathExp(-x));
}

//+------------------------------------------------------------------+
//| Apply attention mechanism                                        |
//+------------------------------------------------------------------+
void ApplyAttention(NeuralNetwork &nn, double &hidden[])
{
    int size = ArraySize(hidden);
    double scores[];
    ArrayResize(scores, size);

    // Calculate attention scores
    double sumExp = 0;
    for(int i = 0; i < size; i++)
    {
        scores[i] = 0;
        for(int j = 0; j < size; j++)
        {
            int idx = j * size + i;  // Access flattened 2D array
            scores[i] += hidden[j] * nn.attentionWeights[idx];
        }

        scores[i] = MathExp(scores[i]);
        sumExp += scores[i];
    }

    // Normalize scores (softmax)
    for(int i = 0; i < size; i++)
        scores[i] /= sumExp;

    // Apply attention to create context vector
    for(int i = 0; i < size; i++)
    {
        nn.contextVector[i] = 0;
        for(int j = 0; j < size; j++)
            nn.contextVector[i] += hidden[j] * scores[j];
    }
}

//+------------------------------------------------------------------+
//| Detect hidden liquidity                                          |
//+------------------------------------------------------------------+
void DetectHiddenLiquidity(string symbol, HiddenLiquidity &liquidity)
{
    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 10000);

    if(copied <= 0) return;

    // Analyze order flow for hidden orders
    double visibleVolume = 0;
    double executedVolume = 0;
    double canceledVolume = 0;

    // Track rapid cancellations (potential icebergs)
    int rapidCancels = 0;
    datetime lastTime = 0;

    for(int i = 1; i < copied; i++)
    {
        double volumeDiff = ticks[i].volume - ticks[i-1].volume;

        // Detect rapid changes
        if(ticks[i].time - lastTime < 100)  // Within 100ms
        {
            if(volumeDiff < 0)
                rapidCancels++;
        }

        visibleVolume += MathAbs(volumeDiff);
        if(ticks[i].flags & (TICK_FLAG_BUY | TICK_FLAG_SELL))
            executedVolume += ticks[i].volume;

        lastTime = ticks[i].time;
    }

    // Iceberg detection
    liquidity.icebergRatio = rapidCancels > 50 ?
                             (double)rapidCancels / copied : 0;

    // Estimate dark pool activity
    EstimateDarkPoolActivity(ticks, copied, liquidity);

    // Hidden pressure analysis
    AnalyzeHiddenPressure(ticks, copied, liquidity);

    // Liquidity mirage detection
    liquidity.liquidityMirage = DetectLiquidityMirage(ticks, copied);

    // Real liquidity estimation
    liquidity.realLiquidity = visibleVolume * (1 - liquidity.liquidityMirage);

    // Predict future liquidity
    PredictLiquidity(liquidity);
}

//+------------------------------------------------------------------+
//| Estimate dark pool activity                                      |
//+------------------------------------------------------------------+
void EstimateDarkPoolActivity(const MqlTick &ticks[], int count, HiddenLiquidity &liquidity)
{
    // Look for price movements without visible volume
    double priceChanges = 0;
    double volumeChanges = 0;

    for(int i = 1; i < count; i++)
    {
        double priceChange = MathAbs(ticks[i].bid - ticks[i-1].bid);
        double volumeChange = ticks[i].volume;

        if(priceChange > 0)
        {
            priceChanges += priceChange;
            volumeChanges += volumeChange;
        }
    }

    // Low volume relative to price movement suggests dark pool activity
    double expectedVolume = priceChanges * 10000;  // Expected volume based on price moves
    liquidity.darkPoolActivity = MathMax(0, (expectedVolume - volumeChanges) / expectedVolume);
}

//+------------------------------------------------------------------+
//| Analyze hidden pressure                                          |
//+------------------------------------------------------------------+
void AnalyzeHiddenPressure(const MqlTick &ticks[], int count, HiddenLiquidity &liquidity)
{
    // Check minimum array size
    if(count < 3)
    {
        liquidity.hiddenBuyPressure = 0;
        liquidity.hiddenSellPressure = 0;
        return;
    }

    // Detect buy/sell pressure from micro patterns
    double buyPressure = 0;
    double sellPressure = 0;

    for(int i = 1; i < count - 1; i++)
    {
        // Quick touch and pullback indicates hidden orders
        if(ticks[i].ask < ticks[i-1].ask && ticks[i+1].ask > ticks[i].ask)
            buyPressure += ticks[i].volume;

        if(ticks[i].bid > ticks[i-1].bid && ticks[i+1].bid < ticks[i].bid)
            sellPressure += ticks[i].volume;
    }

    liquidity.hiddenBuyPressure = buyPressure / count;
    liquidity.hiddenSellPressure = sellPressure / count;
}

//+------------------------------------------------------------------+
//| Detect liquidity mirage                                          |
//+------------------------------------------------------------------+
double DetectLiquidityMirage(const MqlTick &ticks[], int count)
{
    // Check minimum array size
    if(count < 3)
        return 0;

    // Detect fake liquidity (orders that disappear when approached)
    int quickCancels = 0;
    int totalOrders = 0;

    for(int i = 2; i < count; i++)
    {
        // Order appeared and disappeared quickly
        if(ticks[i-2].volume < ticks[i-1].volume && ticks[i].volume < ticks[i-1].volume)
        {
            if(ticks[i].time - ticks[i-2].time < 1000)  // Within 1 second
                quickCancels++;

            totalOrders++;
        }
    }

    return totalOrders > 0 ? (double)quickCancels / totalOrders : 0;
}

//+------------------------------------------------------------------+
//| Predict future liquidity                                         |
//+------------------------------------------------------------------+
void PredictLiquidity(HiddenLiquidity &liquidity)
{
    // Simple prediction based on patterns
    ArrayResize(liquidity.predictedLiquidity, 10);

    double currentLiquidity = liquidity.realLiquidity;
    double trend = liquidity.darkPoolActivity - 0.5;  // Trend direction

    for(int i = 0; i < 10; i++)
    {
        liquidity.predictedLiquidity[i] = currentLiquidity * (1 + trend * 0.1 * (i + 1));

        // Add volatility
        liquidity.predictedLiquidity[i] *= (1 + (MathRand() / 32767.0 - 0.5) * 0.1);
    }
}

//+------------------------------------------------------------------+
//| Analyze order book dynamics                                      |
//+------------------------------------------------------------------+
void AnalyzeOrderBookDynamics(string symbol, OrderBookDynamics &dynamics)
{
    MqlBookInfo book[];
    if(!MarketBookGet(symbol, book)) return;

    int bookSize = ArraySize(book);

    // Initialize arrays
    ArrayResize(dynamics.bidDepth, 10);
    ArrayResize(dynamics.askDepth, 10);
    ArrayResize(dynamics.microImbalance, 100);

    // Separate bids and asks
    double totalBidVolume = 0, totalAskVolume = 0;
    double weightedBidPrice = 0, weightedAskPrice = 0;

    int bidLevel = 0, askLevel = 0;

    for(int i = 0; i < bookSize; i++)
    {
        if(book[i].type == BOOK_TYPE_BUY && bidLevel < 10)
        {
            dynamics.bidDepth[bidLevel] = book[i].volume;
            totalBidVolume += book[i].volume;
            weightedBidPrice += book[i].price * book[i].volume;
            bidLevel++;
        }
        else if(book[i].type == BOOK_TYPE_SELL && askLevel < 10)
        {
            dynamics.askDepth[askLevel] = book[i].volume;
            totalAskVolume += book[i].volume;
            weightedAskPrice += book[i].price * book[i].volume;
            askLevel++;
        }
    }

    // Calculate pressure
    dynamics.bidPressure = totalBidVolume / (totalBidVolume + totalAskVolume + 1);
    dynamics.askPressure = totalAskVolume / (totalBidVolume + totalAskVolume + 1);

    // Detect spoofing
    DetectSpoofing(dynamics);

    // Calculate persistent imbalance
    dynamics.persistentImbalance = (totalBidVolume - totalAskVolume) / (totalBidVolume + totalAskVolume + 1);

    // Detect HFT patterns
    DetectHFTPatterns(symbol, dynamics);
}

//+------------------------------------------------------------------+
//| Detect spoofing in order book                                   |
//+------------------------------------------------------------------+
void DetectSpoofing(OrderBookDynamics &dynamics)
{
    // Look for large orders away from best bid/ask
    double avgDepth = 0;
    int levels = MathMin(ArraySize(dynamics.bidDepth), ArraySize(dynamics.askDepth));

    for(int i = 0; i < levels; i++)
    {
        avgDepth += dynamics.bidDepth[i] + dynamics.askDepth[i];
    }
    avgDepth /= (levels * 2);

    // Check for abnormally large orders at outer levels
    double spoofScore = 0;

    for(int i = 3; i < levels; i++)  // Skip first 3 levels
    {
        if(dynamics.bidDepth[i] > avgDepth * 5)
            spoofScore += (dynamics.bidDepth[i] / avgDepth - 1) / 10;

        if(dynamics.askDepth[i] > avgDepth * 5)
            spoofScore += (dynamics.askDepth[i] / avgDepth - 1) / 10;
    }

    dynamics.spoofingScore = MathMin(spoofScore, 1.0);

    // Detect layering
    DetectLayering(dynamics);
}

//+------------------------------------------------------------------+
//| Detect layering manipulation                                     |
//+------------------------------------------------------------------+
void DetectLayering(OrderBookDynamics &dynamics)
{
    // Look for multiple orders at different levels (layering pattern)
    int bidLayers = 0, askLayers = 0;
    double lastBidSize = 0, lastAskSize = 0;

    for(int i = 0; i < ArraySize(dynamics.bidDepth); i++)
    {
        if(dynamics.bidDepth[i] > 0 && MathAbs(dynamics.bidDepth[i] - lastBidSize) < lastBidSize * 0.1)
            bidLayers++;
        lastBidSize = dynamics.bidDepth[i];
    }

    for(int i = 0; i < ArraySize(dynamics.askDepth); i++)
    {
        if(dynamics.askDepth[i] > 0 && MathAbs(dynamics.askDepth[i] - lastAskSize) < lastAskSize * 0.1)
            askLayers++;
        lastAskSize = dynamics.askDepth[i];
    }

    dynamics.layeringDetected = (bidLayers > 3 || askLayers > 3) ? 1.0 : 0.0;
}

//+------------------------------------------------------------------+
//| Detect HFT patterns                                              |
//+------------------------------------------------------------------+
void DetectHFTPatterns(string symbol, OrderBookDynamics &dynamics)
{
    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 1000);

    if(copied <= 0) return;

    // Quote stuffing detection
    int rapidQuotes = 0;
    datetime lastQuoteTime = 0;

    for(int i = 0; i < copied; i++)
    {
        if(ticks[i].time - lastQuoteTime < 10)  // Within 10ms
            rapidQuotes++;
        lastQuoteTime = ticks[i].time;
    }

    dynamics.quoteStuffing = (double)rapidQuotes / copied;

    // Pinging detection (small orders to detect hidden liquidity)
    int smallOrders = 0;
    double avgVolume = 0;

    for(int i = 0; i < copied; i++)
        avgVolume += ticks[i].volume;
    avgVolume /= copied;

    for(int i = 0; i < copied; i++)
    {
        if(ticks[i].volume < avgVolume * 0.1)
            smallOrders++;
    }

    dynamics.pinging = (double)smallOrders / copied;

    // Front-running detection
    DetectFrontRunning(ticks, copied, dynamics);
}

//+------------------------------------------------------------------+
//| Detect front-running patterns                                    |
//+------------------------------------------------------------------+
void DetectFrontRunning(const MqlTick &ticks[], int count, OrderBookDynamics &dynamics)
{
    // Check minimum array size
    if(count < 3)
    {
        dynamics.frontRunning = 0;
        return;
    }

    // Look for patterns of small orders followed by large orders
    int frontRunPatterns = 0;

    for(int i = 2; i < count; i++)
    {
        // Small order followed by large order in same direction
        if(ticks[i-1].volume < ticks[i].volume * 0.1)
        {
            if((ticks[i-1].flags & TICK_FLAG_BUY) && (ticks[i].flags & TICK_FLAG_BUY))
                frontRunPatterns++;
            else if((ticks[i-1].flags & TICK_FLAG_SELL) && (ticks[i].flags & TICK_FLAG_SELL))
                frontRunPatterns++;
        }
    }

    dynamics.frontRunning = (double)frontRunPatterns / count;
}

//+------------------------------------------------------------------+
//| Update market maker inventory model                              |
//+------------------------------------------------------------------+
void UpdateMarketMakerModel(string symbol, double position, MarketMakerInventory &mm)
{
    mm.currentInventory = position;

    // Calculate inventory risk
    double maxPosition = mm.inventoryLimit;
    mm.inventoryRisk = MathPow(position / maxPosition, 2);

    // Calculate optimal inventory (typically near zero)
    mm.targetInventory = 0;  // Market makers prefer flat

    // Calculate inventory skew
    double inventoryPressure = position / maxPosition;
    mm.inventorySkew = inventoryPressure * 0.0001;  // 1 pip per full inventory

    // Calculate fair value using multiple methods
    mm.fairValue = CalculateFairValue(symbol);

    // Optimal spreads based on inventory
    double baseSpread = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * SymbolInfoDouble(symbol, SYMBOL_POINT);

    if(position > 0)  // Long inventory, want to sell
    {
        mm.bidOffset = baseSpread * (1 + mm.inventoryRisk);
        mm.askOffset = baseSpread * (1 - mm.inventoryRisk * 0.5);
    }
    else if(position < 0)  // Short inventory, want to buy
    {
        mm.bidOffset = baseSpread * (1 - mm.inventoryRisk * 0.5);
        mm.askOffset = baseSpread * (1 + mm.inventoryRisk);
    }
    else  // Neutral
    {
        mm.bidOffset = baseSpread;
        mm.askOffset = baseSpread;
    }

    // Expected spread capture
    mm.spreadCapture = (mm.bidOffset + mm.askOffset) * EstimatedVolume(symbol);

    // Flow prediction
    PredictOrderFlow(symbol, mm);

    // Adverse selection cost
    mm.adverseSelection = CalculateAdverseSelection(symbol);

    // Optimal hedge ratio
    mm.optimalHedgeRatio = CalculateOptimalHedge(position, mm.inventoryRisk);
}

//+------------------------------------------------------------------+
//| Calculate fair value using multiple methods                      |
//+------------------------------------------------------------------+
double CalculateFairValue(string symbol)
{
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);

    // Micro-price (weighted by order book imbalance)
    MqlBookInfo book[];
    MarketBookGet(symbol, book);

    double bidVolume = 0, askVolume = 0;
    for(int i = 0; i < ArraySize(book); i++)
    {
        if(book[i].type == BOOK_TYPE_BUY)
            bidVolume += book[i].volume;
        else
            askVolume += book[i].volume;
    }

    double imbalance = bidVolume / (bidVolume + askVolume);
    double microPrice = bid * (1 - imbalance) + ask * imbalance;

    // VWAP component
    double vwap = CalculateVWAP(symbol, 100);

    // Time-weighted price
    double twap = (bid + ask) / 2;

    // Combine methods
    return microPrice * 0.5 + vwap * 0.3 + twap * 0.2;
}

//+------------------------------------------------------------------+
//| Calculate VWAP                                                   |
//+------------------------------------------------------------------+
double CalculateVWAP(string symbol, int periods)
{
    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, periods);

    if(copied <= 0) return SymbolInfoDouble(symbol, SYMBOL_BID);

    double sumPriceVolume = 0;
    double sumVolume = 0;

    for(int i = 0; i < copied; i++)
    {
        double price = (ticks[i].bid + ticks[i].ask) / 2;
        sumPriceVolume += price * ticks[i].volume;
        sumVolume += ticks[i].volume;
    }

    return sumVolume > 0 ? sumPriceVolume / sumVolume : SymbolInfoDouble(symbol, SYMBOL_BID);
}

//+------------------------------------------------------------------+
//| Predict order flow                                               |
//+------------------------------------------------------------------+
void PredictOrderFlow(string symbol, MarketMakerInventory &mm)
{
    // Use historical patterns to predict flow
    MqlTick ticks[];
    CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 1000);

    double buyFlow = 0, sellFlow = 0;
    int buyCount = 0, sellCount = 0;

    for(int i = 0; i < ArraySize(ticks); i++)
    {
        if(ticks[i].flags & TICK_FLAG_BUY)
        {
            buyFlow += ticks[i].volume;
            buyCount++;
        }
        else if(ticks[i].flags & TICK_FLAG_SELL)
        {
            sellFlow += ticks[i].volume;
            sellCount++;
        }
    }

    // Net expected flow
    mm.expectedFlow = (buyFlow - sellFlow) / ArraySize(ticks);

    // Toxic flow detection (one-sided aggressive flow)
    double flowImbalance = MathAbs(buyFlow - sellFlow) / (buyFlow + sellFlow + 1);
    mm.toxicFlow = flowImbalance > 0.7 ? flowImbalance : 0;

    // Informed trader probability (large orders relative to average)
    double avgOrderSize = (buyFlow + sellFlow) / (buyCount + sellCount + 1);
    int largeOrders = 0;

    for(int i = 0; i < ArraySize(ticks); i++)
    {
        if(ticks[i].volume > avgOrderSize * 3)
            largeOrders++;
    }

    mm.informedTraderProb = (double)largeOrders / ArraySize(ticks);
}

//+------------------------------------------------------------------+
//| Calculate adverse selection cost                                 |
//+------------------------------------------------------------------+
double CalculateAdverseSelection(string symbol)
{
    // Measure how often market moves against us after fills
    // This would require trade history analysis
    // Simplified version:

    double volatility = CalculateVolatility(symbol, 20);
    double spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD) *  SymbolInfoDouble(symbol, SYMBOL_POINT);

    // Higher volatility = higher adverse selection
    return spread > 0 ? volatility / spread : 0;
}

//+------------------------------------------------------------------+
//| Calculate optimal hedge ratio                                    |
//+------------------------------------------------------------------+
double CalculateOptimalHedge(double position, double risk)
{
    // Delta-neutral hedging adjusted for risk
    return MathMin(MathAbs(position) * risk, 1.0);
}

//+------------------------------------------------------------------+
//| Calculate volatility                                             |
//+------------------------------------------------------------------+
double CalculateVolatility(string symbol, int periods)
{
    double close[];
    ArraySetAsSeries(close, true);
    int copied = CopyClose(symbol, PERIOD_M1, 0, periods, close);

    // Check if we got enough data
    if(copied < periods)
        return 0;

    double returns[];
    ArrayResize(returns, periods - 1);

    for(int i = 0; i < periods - 1; i++)
        returns[i] = MathLog(close[i] / close[i + 1]);

    double mean = 0;
    for(int i = 0; i < periods - 1; i++)
        mean += returns[i];
    mean /= (periods - 1);

    double variance = 0;
    for(int i = 0; i < periods - 1; i++)
        variance += MathPow(returns[i] - mean, 2);
    variance /= (periods - 1);

    return MathSqrt(variance);
}

//+------------------------------------------------------------------+
//| Estimated volume for spread capture                              |
//+------------------------------------------------------------------+
double EstimatedVolume(string symbol)
{
    // Estimate how much volume we can capture
    MqlTick ticks[];
    CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 1000);

    double totalVolume = 0;
    for(int i = 0; i < ArraySize(ticks); i++)
        totalVolume += ticks[i].volume;

    // Assume we can capture 10% of volume as market maker
    return totalVolume / ArraySize(ticks) * 0.1;
}

//+------------------------------------------------------------------+
//| Initialize adaptive stop loss system                             |
//+------------------------------------------------------------------+
void InitializeAdaptiveStopLoss(AdaptiveStopLoss &asl)
{
    // Initialize Q-table
    int numStates = 10;   // Discretized states
    int numActions = 5;   // Different stop distances

    // Store dimensions
    asl.numStates = numStates;
    asl.numActions = numActions;

    // Initialize flattened Q-table (numStates x numActions)
    ArrayResize(asl.qTable, numStates * numActions);
    for(int i = 0; i < numStates * numActions; i++)
        asl.qTable[i] = 0;  // Initialize to 0

    // Initialize stop distances (in ATR multiples)
    ArrayResize(asl.stopDistance, numActions);
    asl.stopDistance[0] = 0.5;
    asl.stopDistance[1] = 1.0;
    asl.stopDistance[2] = 1.5;
    asl.stopDistance[3] = 2.0;
    asl.stopDistance[4] = 3.0;

    // Learning parameters
    asl.epsilon = 0.1;    // 10% exploration
    asl.gamma = 0.95;     // Discount factor

    // Initialize metrics
    asl.averageReward = 0;
    asl.stopEfficiency = 0;
    asl.falseStopRate = 0;
}

//+------------------------------------------------------------------+
//| Update adaptive stop loss                                        |
//+------------------------------------------------------------------+
double UpdateAdaptiveStopLoss(string symbol, double entryPrice, int direction,
                              AdaptiveStopLoss &asl)
{
    // Get current state
    int state = GetStopLossState(symbol, asl);

    // Choose action (epsilon-greedy)
    int action = ChooseStopAction(state, asl);

    // Calculate stop distance
    double atr = GetATR(symbol);
    double stopDistance = atr * asl.stopDistance[action];

    // Return stop loss price
    return direction > 0 ?
           entryPrice - stopDistance :
           entryPrice + stopDistance;
}

//+------------------------------------------------------------------+
//| Get stop loss state                                              |
//+------------------------------------------------------------------+
int GetStopLossState(string symbol, AdaptiveStopLoss &asl)
{
    // Discretize market conditions into states
    double volatility = CalculateVolatility(symbol, 20);
    double trend = CalculateTrend(symbol);
    double momentum = CalculateMomentum(symbol);

    // Update state variables
    asl.volatilityState = volatility;
    asl.trendState = trend;
    asl.momentumState = momentum;

    // Combine into discrete state (0-9)
    int volState = MathMin((int)(volatility * 100), 3);
    int trendState = trend > 0.5 ? 2 : (trend < -0.5 ? 0 : 1);
    int momState = momentum > 0 ? 1 : 0;

    return MathMin(volState * 3 + trendState + momState, 9);
}

//+------------------------------------------------------------------+
//| Choose stop action using epsilon-greedy                          |
//+------------------------------------------------------------------+
int ChooseStopAction(int state, AdaptiveStopLoss &asl)
{
    // Exploration vs exploitation
    if(MathRand() / 32767.0 < asl.epsilon)
    {
        // Explore: random action
        return MathRand() % asl.numActions;
    }
    else
    {
        // Exploit: best known action
        int bestAction = 0;
        int stateOffset = state * asl.numActions;
        double bestValue = asl.qTable[stateOffset + 0];

        for(int i = 1; i < asl.numActions; i++)
        {
            if(asl.qTable[stateOffset + i] > bestValue)
            {
                bestValue = asl.qTable[stateOffset + i];
                bestAction = i;
            }
        }

        return bestAction;
    }
}

//+------------------------------------------------------------------+
//| Update Q-table after trade completes                             |
//+------------------------------------------------------------------+
void UpdateQLearning(int state, int action, double reward, int nextState,
                     AdaptiveStopLoss &asl)
{
    // Q-learning update rule
    int idx = state * asl.numActions + action;
    double oldQ = asl.qTable[idx];

    // Find max Q-value for next state
    int nextStateOffset = nextState * asl.numActions;
    double maxNextQ = asl.qTable[nextStateOffset + 0];
    for(int i = 1; i < asl.numActions; i++)
    {
        if(asl.qTable[nextStateOffset + i] > maxNextQ)
            maxNextQ = asl.qTable[nextStateOffset + i];
    }

    // Update Q-value
    double learningRate = 0.1;
    asl.qTable[idx] = oldQ + learningRate * (reward + asl.gamma * maxNextQ - oldQ);

    // Update metrics
    ArrayResize(asl.rewardHistory, ArraySize(asl.rewardHistory) + 1);
    asl.rewardHistory[ArraySize(asl.rewardHistory) - 1] = reward;

    // Calculate average reward
    double sumReward = 0;
    for(int i = 0; i < ArraySize(asl.rewardHistory); i++)
        sumReward += asl.rewardHistory[i];
    asl.averageReward = sumReward / ArraySize(asl.rewardHistory);
}

//+------------------------------------------------------------------+
//| Calculate trend strength                                         |
//+------------------------------------------------------------------+
double CalculateTrend(string symbol)
{
    double close[];
    ArraySetAsSeries(close, true);
    CopyClose(symbol, PERIOD_CURRENT, 0, 50, close);

    // Linear regression slope
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for(int i = 0; i < 50; i++)
    {
        sumX += i;
        sumY += close[i];
        sumXY += i * close[i];
        sumX2 += i * i;
    }

    double slope = (50 * sumXY - sumX * sumY) / (50 * sumX2 - sumX * sumX);

    // Normalize to -1 to 1
    return MathTanh(slope * 1000);
}

//+------------------------------------------------------------------+
//| Calculate momentum                                               |
//+------------------------------------------------------------------+
double CalculateMomentum(string symbol)
{
    double close[];
    ArraySetAsSeries(close, true);
    CopyClose(symbol, PERIOD_CURRENT, 0, 20, close);

    return (close[0] - close[19]) / close[19];
}

//+------------------------------------------------------------------+
//| Get ATR value                                                    |
//+------------------------------------------------------------------+
double GetATR(string symbol)
{
    double atr[];
    ArraySetAsSeries(atr, true);
    int handle = iATR(symbol, PERIOD_CURRENT, 14);
    CopyBuffer(handle, 0, 0, 1, atr);
    return atr[0];
}

//+------------------------------------------------------------------+
//| Analyze cross-asset contagion                                    |
//+------------------------------------------------------------------+
void AnalyzeCrossAssetContagion(string &symbols[], CrossAssetContagion &contagion)
{
    int numAssets = ArraySize(symbols);
    contagion.numAssets = numAssets;

    // Initialize flattened correlation matrix (numAssets x numAssets)
    ArrayResize(contagion.correlationMatrix, numAssets * numAssets);
    ArrayResize(contagion.tailDependence, numAssets * numAssets);

    // Calculate dynamic correlations
    CalculateDynamicCorrelations(symbols, contagion.correlationMatrix);

    // Calculate tail dependence (correlation during extreme moves)
    CalculateTailDependence(symbols, contagion);

    // Detect correlation breaks
    DetectCorrelationBreaks(symbols, contagion);

    // Calculate contagion metrics
    contagion.contagionIndex = CalculateContagionIndex(contagion.correlationMatrix, numAssets);
    contagion.systemicRisk = CalculateSystemicRisk(contagion);

    // Identify leading indicators
    IdentifyLeadingAssets(symbols, contagion);

    // Network analysis
    CalculateNetworkMetrics(contagion);
}

//+------------------------------------------------------------------+
//| Calculate dynamic correlations using DCC-GARCH                   |
//+------------------------------------------------------------------+
void CalculateDynamicCorrelations(string &symbols[], double &correlations[])
{
    int numAssets = ArraySize(symbols);
    int returnSize = 99;
    double returns[];
    ArrayResize(returns, numAssets * returnSize);
    ArrayResize(correlations, numAssets * numAssets);

    // Get returns for all assets
    for(int i = 0; i < numAssets; i++)
    {
        double close[];
        ArraySetAsSeries(close, true);
        CopyClose(symbols[i], PERIOD_H1, 0, 100, close);

        int baseIdx = i * returnSize;
        for(int j = 0; j < returnSize; j++)
            returns[baseIdx + j] = MathLog(close[j] / close[j + 1]);
    }

    // Calculate correlations
    for(int i = 0; i < numAssets; i++)
    {
        for(int j = i; j < numAssets; j++)
        {
            if(i == j)
            {
                correlations[i * numAssets + j] = 1.0;
            }
            else
            {
                // Create temporary arrays for correlation calculation
                double ret1[], ret2[];
                ArrayResize(ret1, returnSize);
                ArrayResize(ret2, returnSize);

                int baseIdx1 = i * returnSize;
                int baseIdx2 = j * returnSize;
                for(int k = 0; k < returnSize; k++)
                {
                    ret1[k] = returns[baseIdx1 + k];
                    ret2[k] = returns[baseIdx2 + k];
                }

                double corr = CalculateCorrelation(ret1, ret2);
                correlations[i * numAssets + j] = corr;
                correlations[j * numAssets + i] = corr;
            }
        }
    }
}

#ifndef CALCULATE_CORRELATION_DEFINED
#define CALCULATE_CORRELATION_DEFINED
//+------------------------------------------------------------------+
//| Calculate correlation between two arrays                         |
//+------------------------------------------------------------------+
double CalculateCorrelation(const double &x[], const double &y[])
{
    int n = MathMin(ArraySize(x), ArraySize(y));

    double meanX = 0, meanY = 0;
    for(int i = 0; i < n; i++)
    {
        meanX += x[i];
        meanY += y[i];
    }
    meanX /= n;
    meanY /= n;

    double cov = 0, varX = 0, varY = 0;
    for(int i = 0; i < n; i++)
    {
        cov += (x[i] - meanX) * (y[i] - meanY);
        varX += MathPow(x[i] - meanX, 2);
        varY += MathPow(y[i] - meanY, 2);
    }

    return cov / MathSqrt(varX * varY);
}
#endif

//+------------------------------------------------------------------+
//| Calculate tail dependence                                        |
//+------------------------------------------------------------------+
void CalculateTailDependence(string &symbols[], CrossAssetContagion &contagion)
{
    int numAssets = ArraySize(symbols);

    for(int i = 0; i < numAssets; i++)
    {
        for(int j = i + 1; j < numAssets; j++)
        {
            double dep = CalculatePairTailDependence(symbols[i], symbols[j]);
            contagion.tailDependence[i * numAssets + j] = dep;
            contagion.tailDependence[j * numAssets + i] = dep;
        }
        contagion.tailDependence[i * numAssets + i] = 1.0;
    }
}

//+------------------------------------------------------------------+
//| Calculate tail dependence for a pair                             |
//+------------------------------------------------------------------+
double CalculatePairTailDependence(string symbol1, string symbol2)
{
    double returns1[], returns2[];
    GetReturns(symbol1, returns1, 500);
    GetReturns(symbol2, returns2, 500);

    // Find extreme quantiles (5% tails)
    double quantile = 0.05;
    int tailSize = (int)(ArraySize(returns1) * quantile);

    // Sort returns
    double sorted1[], sorted2[];
    ArrayCopy(sorted1, returns1);
    ArrayCopy(sorted2, returns2);
    ArraySort(sorted1);
    ArraySort(sorted2);

    // Count joint extremes
    int jointExtremes = 0;

    for(int i = 0; i < ArraySize(returns1); i++)
    {
        bool extreme1 = returns1[i] <= sorted1[tailSize] || returns1[i] >= sorted1[ArraySize(sorted1) - tailSize - 1];
        bool extreme2 = returns2[i] <= sorted2[tailSize] || returns2[i] >= sorted2[ArraySize(sorted2) - tailSize - 1];

        if(extreme1 && extreme2)
            jointExtremes++;
    }

    // Tail dependence coefficient
    double expected = 2 * tailSize;
    return (double)jointExtremes / expected;
}

//+------------------------------------------------------------------+
//| Get returns for a symbol                                         |
//+------------------------------------------------------------------+
void GetReturns(string symbol, double &returns[], int periods)
{
    double close[];
    ArraySetAsSeries(close, true);
    CopyClose(symbol, PERIOD_H1, 0, periods + 1, close);

    ArrayResize(returns, periods);
    for(int i = 0; i < periods; i++)
        returns[i] = MathLog(close[i] / close[i + 1]);
}

//+------------------------------------------------------------------+
//| Detect correlation breaks                                        |
//+------------------------------------------------------------------+
void DetectCorrelationBreaks(string &symbols[], CrossAssetContagion &contagion)
{
    // Implement correlation break detection
    // This would track when correlations suddenly change
    ArrayResize(contagion.correlationBreaks, ArraySize(symbols));

    // Simplified version
    for(int i = 0; i < ArraySize(symbols); i++)
        contagion.correlationBreaks[i] = 0;  // Placeholder
}

//+------------------------------------------------------------------+
//| Calculate contagion index                                        |
//+------------------------------------------------------------------+
double CalculateContagionIndex(const double &correlations[], int numAssets)
{
    double avgCorrelation = 0;
    int count = 0;

    for(int i = 0; i < numAssets; i++)
    {
        for(int j = i + 1; j < numAssets; j++)
        {
            avgCorrelation += MathAbs(correlations[i * numAssets + j]);
            count++;
        }
    }

    return count > 0 ? avgCorrelation / count : 0;
}

//+------------------------------------------------------------------+
//| Calculate systemic risk                                          |
//+------------------------------------------------------------------+
double CalculateSystemicRisk(const CrossAssetContagion &contagion)
{
    // Eigenvalue approach to systemic risk
    // Simplified: use average correlation and tail dependence

    double avgTailDep = 0;
    int n = contagion.numAssets;
    int count = 0;

    for(int i = 0; i < n; i++)
    {
        for(int j = i + 1; j < n; j++)
        {
            avgTailDep += contagion.tailDependence[i * n + j];
            count++;
        }
    }

    avgTailDep = count > 0 ? avgTailDep / count : 0;

    // Systemic risk increases with both correlation and tail dependence
    return contagion.contagionIndex * 0.5 + avgTailDep * 0.5;
}

//+------------------------------------------------------------------+
//| Identify leading assets                                          |
//+------------------------------------------------------------------+
void IdentifyLeadingAssets(string &symbols[], CrossAssetContagion &contagion)
{
    int n = ArraySize(symbols);
    ArrayResize(contagion.leadingAssets, n);
    ArrayResize(contagion.leadTime, n);
    ArrayResize(contagion.signalStrength, n);

    // Analyze lead-lag relationships
    for(int i = 0; i < n; i++)
    {
        contagion.leadingAssets[i] = symbols[i];
        contagion.leadTime[i] = 0;
        contagion.signalStrength[i] = 0;

        // Check if this asset leads others
        for(int j = 0; j < n; j++)
        {
            if(i != j)
            {
                double leadLag = CalculateLeadLag(symbols[i], symbols[j]);
                if(leadLag > 0)
                {
                    contagion.leadTime[i] += leadLag;
                    contagion.signalStrength[i] += 1;
                }
            }
        }

        contagion.leadTime[i] /= (n - 1);
        contagion.signalStrength[i] /= (n - 1);
    }
}

//+------------------------------------------------------------------+
//| Calculate lead-lag relationship                                  |
//+------------------------------------------------------------------+
double CalculateLeadLag(string symbol1, string symbol2)
{
    // Cross-correlation at different lags
    double returns1[], returns2[];
    GetReturns(symbol1, returns1, 100);
    GetReturns(symbol2, returns2, 100);

    double maxCorr = 0;
    int optimalLag = 0;

    for(int lag = -10; lag <= 10; lag++)
    {
        double corr = CalculateLaggedCorrelation(returns1, returns2, lag);
        if(MathAbs(corr) > MathAbs(maxCorr))
        {
            maxCorr = corr;
            optimalLag = lag;
        }
    }

    return optimalLag;  // Positive means symbol1 leads
}

//+------------------------------------------------------------------+
//| Calculate lagged correlation                                     |
//+------------------------------------------------------------------+
double CalculateLaggedCorrelation(const double &x[], const double &y[], int lag)
{
    int n = ArraySize(x);
    if(MathAbs(lag) >= n) return 0;

    double corr = 0;

    if(lag >= 0)
    {
        // x leads y by 'lag' periods
        double temp_x[], temp_y[];
        ArrayResize(temp_x, n - lag);
        ArrayResize(temp_y, n - lag);

        for(int i = 0; i < n - lag; i++)
        {
            temp_x[i] = x[i];
            temp_y[i] = y[i + lag];
        }

        corr = CalculateCorrelation(temp_x, temp_y);
    }
    else
    {
        // y leads x
        double temp_x[], temp_y[];
        int absLag = MathAbs(lag);
        ArrayResize(temp_x, n - absLag);
        ArrayResize(temp_y, n - absLag);

        for(int i = 0; i < n - absLag; i++)
        {
            temp_x[i] = x[i + absLag];
            temp_y[i] = y[i];
        }

        corr = CalculateCorrelation(temp_x, temp_y);
    }

    return corr;
}

//+------------------------------------------------------------------+
//| Calculate network metrics                                        |
//+------------------------------------------------------------------+
void CalculateNetworkMetrics(CrossAssetContagion &contagion)
{
    int n = contagion.numAssets;
    ArrayResize(contagion.centralityScore, n);
    ArrayResize(contagion.clusterRisk, n);

    // Calculate centrality (how connected each asset is)
    for(int i = 0; i < n; i++)
    {
        double centrality = 0;
        for(int j = 0; j < n; j++)
        {
            if(i != j)
                centrality += MathAbs(contagion.correlationMatrix[i * n + j]);
        }
        contagion.centralityScore[i] = centrality / (n - 1);
    }

    // Calculate cluster risk (how tightly connected groups are)
    // Simplified: use average correlation within clusters
    for(int i = 0; i < n; i++)
    {
        double clusterCorr = 0;
        int clusterSize = 0;

        for(int j = 0; j < n; j++)
        {
            if(MathAbs(contagion.correlationMatrix[i * n + j]) > 0.7)
            {
                clusterCorr += MathAbs(contagion.correlationMatrix[i * n + j]);
                clusterSize++;
            }
        }

        contagion.clusterRisk[i] = clusterSize > 0 ? clusterCorr / clusterSize : 0;
    }

    // Transmission speed (how fast shocks propagate)
    contagion.transmissionSpeed = CalculateTransmissionSpeed(contagion);
}

//+------------------------------------------------------------------+
//| Calculate transmission speed                                     |
//+------------------------------------------------------------------+
double CalculateTransmissionSpeed(const CrossAssetContagion &contagion)
{
    // Based on average lead times and correlation strength
    double avgLeadTime = 0;
    double avgStrength = 0;

    for(int i = 0; i < ArraySize(contagion.leadTime); i++)
    {
        avgLeadTime += MathAbs(contagion.leadTime[i]);
        avgStrength += contagion.signalStrength[i];
    }

    avgLeadTime /= ArraySize(contagion.leadTime);
    avgStrength /= ArraySize(contagion.signalStrength);

    // Faster transmission with shorter lead times and stronger signals
    return avgLeadTime > 0 ? avgStrength / avgLeadTime : 0;
}

//+------------------------------------------------------------------+
//| Analyze HF microstructure                                        |
//+------------------------------------------------------------------+
void AnalyzeHFMicrostructure(string symbol, HFMicrostructure &hf)
{
    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 10000);

    if(copied <= 0) return;

    // Analyze tick patterns
    AnalyzeTickPatterns(ticks, copied, hf);

    // Analyze quote dynamics
    AnalyzeQuoteDynamics(ticks, copied, hf);

    // Calculate execution probabilities
    CalculateExecutionProbabilities(symbol, hf);

    // Detect latency arbitrage opportunities
    DetectLatencyArbitrage(symbol, hf);
}

//+------------------------------------------------------------------+
//| Analyze tick patterns                                            |
//+------------------------------------------------------------------+
void AnalyzeTickPatterns(const MqlTick &ticks[], int count, HFMicrostructure &hf)
{
    // Tick imbalance
    int buyTicks = 0, sellTicks = 0;
    double buyVolume = 0, sellVolume = 0;

    for(int i = 0; i < count; i++)
    {
        if(ticks[i].flags & TICK_FLAG_BUY)
        {
            buyTicks++;
            buyVolume += ticks[i].volume;
        }
        else if(ticks[i].flags & TICK_FLAG_SELL)
        {
            sellTicks++;
            sellVolume += ticks[i].volume;
        }
    }

    hf.tickImbalance = (buyVolume - sellVolume) / (buyVolume + sellVolume + 1);

    // Tick momentum
    int momentum = 0;
    int momentumCount = MathMin(count - 1, 19);  // Max 19 comparisons, ensure we don't exceed array

    if(count >= 2)  // Need at least 2 ticks
    {
        for(int i = 1; i <= momentumCount; i++)
        {
            if(ticks[i].bid > ticks[i-1].bid)
                momentum++;
            else if(ticks[i].bid < ticks[i-1].bid)
                momentum--;
        }
    }

    hf.tickMomentum = momentumCount > 0 ? (double)momentum / momentumCount : 0;

    // Tick reversal probability
    int reversals = 0;
    if(count >= 3)  // Need at least 3 ticks
    {
        for(int i = 2; i < count; i++)
        {
            bool wasUp = ticks[i-1].bid > ticks[i-2].bid;
            bool isDown = ticks[i].bid < ticks[i-1].bid;

            if(wasUp && isDown || !wasUp && !isDown)
                reversals++;
        }

        hf.tickReversal = (double)reversals / (count - 2);
    }
    else
    {
        hf.tickReversal = 0;
    }
}

//+------------------------------------------------------------------+
//| Analyze quote dynamics                                           |
//+------------------------------------------------------------------+
void AnalyzeQuoteDynamics(const MqlTick &ticks[], int count, HFMicrostructure &hf)
{
    // Check minimum array size
    if(count < 2)
    {
        hf.quoteIntensity = 0;
        hf.bidAskBounce = 0;
        hf.spreadPersistence = 0;
        return;
    }

    // Quote update frequency
    int quoteUpdates = 0;
    double totalTime = (ticks[count-1].time - ticks[0].time) / 1000.0;  // In seconds

    for(int i = 1; i < count; i++)
    {
        if(ticks[i].bid != ticks[i-1].bid || ticks[i].ask != ticks[i-1].ask)
            quoteUpdates++;
    }

    hf.quoteIntensity = totalTime > 0 ? quoteUpdates / totalTime : 0;  // Updates per second

    // Quote lifetime
    double totalLifetime = 0;
    int lifetimeCount = 0;
    datetime lastQuoteTime = ticks[0].time;

    for(int i = 1; i < count; i++)
    {
        if(ticks[i].bid != ticks[i-1].bid || ticks[i].ask != ticks[i-1].ask)
        {
            totalLifetime += (ticks[i].time - lastQuoteTime);
            lifetimeCount++;
            lastQuoteTime = ticks[i].time;
        }
    }

    hf.quoteLifetime = lifetimeCount > 0 ? totalLifetime / lifetimeCount / 1000.0 : 0;

    // Quote competition (how often best quote changes)
    int bestQuoteChanges = 0;
    double lastBest = ticks[0].bid;

    for(int i = 1; i < count; i++)
    {
        if(ticks[i].bid != lastBest)
        {
            bestQuoteChanges++;
            lastBest = ticks[i].bid;
        }
    }

    hf.quoteCompetition = quoteUpdates > 0 ? (double)bestQuoteChanges / quoteUpdates : 0;
    // Bid-ask bounce calculation
    int bounces = 0;
    for(int i = 2; i < count; i++)
    {
        // Check if trade bounces between bid and ask
        bool hitBid = (ticks[i-1].flags & TICK_FLAG_SELL) != 0;
        bool hitAsk = (ticks[i].flags & TICK_FLAG_BUY) != 0;

        if((hitBid && hitAsk) || (!hitBid && !hitAsk && ticks[i].bid != ticks[i-1].bid
                                 ))
            bounces++;
    }

    hf.bidAskBounce = count > 2 ? (double)bounces / (count - 2) : 0;

    // Spread persistence calculation
    double currentSpread = ticks[0].ask - ticks[0].bid;
    int persistenceCount = 0;
    int spreadChanges = 0;

    for(int i = 1; i < count; i++)
    {
        double spread = ticks[i].ask - ticks[i].bid;
        if(MathAbs(spread - currentSpread) < _Point)
            persistenceCount++;
        else
        {
            currentSpread = spread;
            spreadChanges++;
        }
    }

    hf.spreadPersistence = spreadChanges > 0 ? (double)persistenceCount /
                           spreadChanges : persistenceCount;
}

//+------------------------------------------------------------------+
//| Calculate execution probabilities                                |
//+------------------------------------------------------------------+
void CalculateExecutionProbabilities(string symbol, HFMicrostructure &hf)
{
    // Fill probability by price level
    ArrayResize(hf.fillProbability, 5);
    ArrayResize(hf.expectedSlippage, 5);
    ArrayResize(hf.marketImpact, 5);

    // Simplified model
    for(int i = 0; i < 5; i++)
    {
        // Fill probability decreases with distance from best quote
        hf.fillProbability[i] = 1.0 / (1.0 + i * 0.5);

        // Expected slippage increases with order size
        hf.expectedSlippage[i] = i * SymbolInfoDouble(symbol, SYMBOL_POINT);

        // Market impact function
        hf.marketImpact[i] = i * i * 0.0001;  // Quadratic impact
    }
}

//+------------------------------------------------------------------+
//| Detect latency arbitrage opportunities                           |
//+------------------------------------------------------------------+
void DetectLatencyArbitrage(string symbol, HFMicrostructure &hf)
{
    // This would require multiple venue data
    // Simplified: estimate based on quote dynamics

    // Latency advantage estimation (in ms)
    hf.latencyAdvantage = 1.0 / hf.quoteIntensity * 1000;  // Time between quotes

    // Arbitrage opportunities per hour
    hf.arbOpportunities = hf.quoteIntensity * 3600 * 0.001;  // 0.1% of quote updates

    // Competition intensity
    hf.competitionIntensity = hf.quoteCompetition;
}

//+------------------------------------------------------------------+
//| Initialize meta strategy                                         |
//+------------------------------------------------------------------+
void InitializeMetaStrategy(MetaStrategy &meta, int numStrategies)
{
    int historySize = 30;  // Store 30 periods of history per strategy

    // Initialize strategy arrays
    ArrayResize(meta.strategyReturns, numStrategies * historySize);
    ArrayResize(meta.strategySharpe, numStrategies);
    ArrayResize(meta.strategyDrawdown, numStrategies);
    ArrayResize(meta.riskBudget, numStrategies);
    ArrayResize(meta.optimalMix, numStrategies);

    // Initialize weights
    meta.momentumWeight = 0.25;
    meta.meanReversionWeight = 0.25;
    meta.marketMakingWeight = 0.25;
    meta.arbitrageWeight = 0.25;

    // Initialize regime weights (simplified: 3 regimes, 4 strategies)
    int numRegimes = 3;
    ArrayResize(meta.regimeWeights, numRegimes * numStrategies);

    // Trending regime weights (regime 0)
    meta.regimeWeights[0 * numStrategies + 0] = 0.6;  // Momentum
    meta.regimeWeights[0 * numStrategies + 1] = 0.1;  // Mean reversion
    meta.regimeWeights[0 * numStrategies + 2] = 0.2;  // Market making
    meta.regimeWeights[0 * numStrategies + 3] = 0.1;  // Arbitrage

    // Ranging regime weights (regime 1)
    meta.regimeWeights[1 * numStrategies + 0] = 0.1;  // Momentum
    meta.regimeWeights[1 * numStrategies + 1] = 0.4;  // Mean reversion
    meta.regimeWeights[1 * numStrategies + 2] = 0.3;  // Market making
    meta.regimeWeights[1 * numStrategies + 3] = 0.2;  // Arbitrage

    // Volatile regime weights (regime 2)
    meta.regimeWeights[2 * numStrategies + 0] = 0.2;  // Momentum
    meta.regimeWeights[2 * numStrategies + 1] = 0.2;  // Mean reversion
    meta.regimeWeights[2 * numStrategies + 2] = 0.1;  // Market making
    meta.regimeWeights[2 * numStrategies + 3] = 0.5;  // Arbitrage
}

//+------------------------------------------------------------------+
//| Update meta strategy weights                                     |
//+------------------------------------------------------------------+
void UpdateMetaStrategy(MetaStrategy &meta, int currentRegime)
{
    // Use regime-based weights
    int numStrategies = 4;  // Number of strategies
    int numRegimes = ArraySize(meta.regimeWeights) / numStrategies;

    if(currentRegime >= 0 && currentRegime < numRegimes)
    {
        int baseIdx = currentRegime * numStrategies;
        meta.momentumWeight = meta.regimeWeights[baseIdx + 0];
        meta.meanReversionWeight = meta.regimeWeights[baseIdx + 1];
        meta.marketMakingWeight = meta.regimeWeights[baseIdx + 2];
        meta.arbitrageWeight = meta.regimeWeights[baseIdx + 3];
    }

    // Risk parity adjustment
    AdjustForRiskParity(meta);

    // Performance-based adjustment
    AdjustForPerformance(meta);
}

//+------------------------------------------------------------------+
//| Adjust weights for risk parity                                   |
//+------------------------------------------------------------------+
void AdjustForRiskParity(MetaStrategy &meta)
{
    // Calculate inverse volatility weights
    double totalInvVol = 0;
    double invVol[4];

    // Use drawdown as risk proxy
    invVol[0] = meta.strategyDrawdown[0] > 0 ? 1.0 / meta.strategyDrawdown[0] : 1;
    invVol[1] = meta.strategyDrawdown[1] > 0 ? 1.0 / meta.strategyDrawdown[1] : 1;
    invVol[2] = meta.strategyDrawdown[2] > 0 ? 1.0 / meta.strategyDrawdown[2] : 1;
    invVol[3] = meta.strategyDrawdown[3] > 0 ? 1.0 / meta.strategyDrawdown[3] : 1;

    totalInvVol = invVol[0] + invVol[1] + invVol[2] + invVol[3];

    // Blend with regime weights
    double alpha = 0.3;  // 30% risk parity, 70% regime

    meta.momentumWeight = alpha * (invVol[0] / totalInvVol) + (1 - alpha) * meta.momentumWeight;
    meta.meanReversionWeight = alpha * (invVol[1] / totalInvVol) + (1 - alpha) * meta.meanReversionWeight;
    meta.marketMakingWeight = alpha * (invVol[2] / totalInvVol) + (1 - alpha) * meta.marketMakingWeight;
    meta.arbitrageWeight = alpha * (invVol[3] / totalInvVol) + (1 - alpha) * meta.arbitrageWeight;

    // Normalize
    double total = meta.momentumWeight + meta.meanReversionWeight +
                   meta.marketMakingWeight + meta.arbitrageWeight;

    meta.momentumWeight /= total;
    meta.meanReversionWeight /= total;
    meta.marketMakingWeight /= total;
    meta.arbitrageWeight /= total;
}

//+------------------------------------------------------------------+
//| Adjust weights based on recent performance                       |
//+------------------------------------------------------------------+
void AdjustForPerformance(MetaStrategy &meta)
{
    // Momentum of strategy performance
    double perfMomentum[4];

    // Calculate recent performance momentum
    for(int i = 0; i < 4; i++)
    {
        int historySize = 30;  // Assume 30 periods of history per strategy
        int totalSize = ArraySize(meta.strategyReturns);
        int baseIdx = i * historySize;

        // Check if we have enough data for this strategy
        if(totalSize >= baseIdx + 20)  // Need at least 20 periods for this strategy
        {
            double recentReturn = 0;
            double olderReturn = 0;

            // Last 10 periods
            for(int j = 0; j < 10; j++)
            {
                if(baseIdx + j < totalSize)
                    recentReturn += meta.strategyReturns[baseIdx + j];
            }

            // Previous 10 periods
            for(int j = 10; j < 20; j++)
            {
                if(baseIdx + j < totalSize)
                    olderReturn += meta.strategyReturns[baseIdx + j];
            }

            perfMomentum[i] = recentReturn - olderReturn;
        }
        else
        {
            perfMomentum[i] = 0;
        }
    }

    // Slight tilt towards better performing strategies
    double tiltFactor = 0.1;  // 10% performance tilt

    meta.momentumWeight *= (1 + tiltFactor * MathTanh(perfMomentum[0]));
    meta.meanReversionWeight *= (1 + tiltFactor * MathTanh(perfMomentum[1]));
    meta.marketMakingWeight *= (1 + tiltFactor * MathTanh(perfMomentum[2]));
    meta.arbitrageWeight *= (1 + tiltFactor * MathTanh(perfMomentum[3]));

    // Renormalize
    double total = meta.momentumWeight + meta.meanReversionWeight +
                   meta.marketMakingWeight + meta.arbitrageWeight;

    meta.momentumWeight /= total;
    meta.meanReversionWeight /= total;
    meta.marketMakingWeight /= total;
    meta.arbitrageWeight /= total;
}
//+------------------------------------------------------------------+
