/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package handwritten.digit.recog.mnsit;


/**
 *
 * @author mortimer
 */

public class HandwrittenDigitRecogMNSIT {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        DataProcessForGitHub.processDataForGitHubFileSizeLimit();
        NeuralNetwork ne=new NeuralNetwork(new int[]{784,128,10});
        double learningRate=0.001;int noOfEpochs=5;int batchSize=128;
        int totalSizeOfTrainingData=60000;int printAfterNoIterations = 468/2;boolean loadTraingingDataInMemory=true;
        ne.trainNew(learningRate, noOfEpochs, batchSize,  totalSizeOfTrainingData,printAfterNoIterations,loadTraingingDataInMemory);
        MenuUI me=new MenuUI();
        me.init(ne.copyNetwork());
        me.setVisible(true);
    }  
}
