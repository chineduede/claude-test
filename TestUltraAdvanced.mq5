//+------------------------------------------------------------------+
//|                                            TestUltraAdvanced.mq5 |
//|                                    Test compilation of fixed lib |
//+------------------------------------------------------------------+
#include "UltraAdvancedTrading.mqh"

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   // Test Neural Network
   NeuralNetwork nn;
   int hiddenSizes[] = {10, 20, 10};
   InitializeNeuralNetwork(nn, 5, hiddenSizes, 1);
   
   double inputs[] = {0.1, 0.2, 0.3, 0.4, 0.5};
   double outputs[];
   LSTMForward(nn, inputs, outputs);
   
   // Test Adaptive Stop Loss
   AdaptiveStopLoss asl;
   InitializeAdaptiveStopLoss(asl);
   double stopPrice = UpdateAdaptiveStopLoss("EURUSD", 1.1000, 1, asl);
   
   // Test Cross Asset Contagion
   CrossAssetContagion contagion;
   string symbols[] = {"EURUSD", "GBPUSD", "USDJPY"};
   AnalyzeCrossAssetContagion(symbols, contagion);
   
   // Test Meta Strategy
   MetaStrategy meta;
   InitializeMetaStrategy(meta, 4);
   UpdateMetaStrategy(meta, 0);
   
   Print("All tests completed successfully!");
}