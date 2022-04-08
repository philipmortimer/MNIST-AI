/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package handwritten.digit.recog.mnsit;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.stream.Stream;
/**
 *
 * @author mortimer
 */
public class NeuralNetwork {
    final int netStructureLength;
    final int netStructureLengthMinusOne;
    final int allLength;
    final int[]totNeuron;//
    final int[]currentAll;//
    final int[]indexForAllWeightAndError;//
    final int[]indexForActivError;//
    final int activIn;
    final int[]netStructure;
    //int indexFirstLayerActiv;
    double[]allWeightsAndBiases;// in the form of bias, all weights leading into bias going down in array is oind down/left
    final int totalNeurons;
    final int totalNeuronsMinusLastLayer;
    double[] preSigActivs;
    //double histOfSquare[];
    double []postSigActivs;
    final String FILE_NAME = "networkInfo.txt";
    final String FILE_NAME_TRAIN="mnist_train.csv";
    final String FILE_NAME_TEST="mnist_test.csv";
    double timeStep;
    double previousMHat[];
    double previousVHat[];
    public  NeuralNetwork(int[]netStructure){
        this.netStructure = netStructure;
        this.netStructureLength=this.netStructure.length;
        this.netStructureLengthMinusOne=this.netStructureLength-1;
        int totalWeightsAndBiases=0;
        int totalNeurons=0;
        for(int i =0;i<netStructure.length;i++){
            totalNeurons=totalNeurons+this.netStructure[i];
            if(i==0){

            }else{
                totalWeightsAndBiases = totalWeightsAndBiases+this.netStructure[i]+(this.netStructure[i]*this.netStructure[i-1]);
            }
        }
        this.totalNeurons = totalNeurons;
        this.totalNeuronsMinusLastLayer=this.totalNeurons-this.netStructure[this.netStructureLengthMinusOne];
        this.allWeightsAndBiases=new double[totalWeightsAndBiases];
        this.allLength=totalWeightsAndBiases;
        this.activIn=this.totalNeurons-this.netStructure[this.netStructure.length-1];
        int currentIndex=0;
        for(int layerIndex=0;layerIndex<this.netStructure.length;layerIndex++){
            if(layerIndex!=0){
                for(int currentLayerIndex=0;currentLayerIndex<this.netStructure[layerIndex];currentLayerIndex++){
                    this.allWeightsAndBiases[currentIndex]=0.0;//Math.random()*0.001;//getRandDoubleBetweenOneAndMinusOne(this.netStructure[layerIndex-1]);//bias in current layer
                    //this.allWeightsAndBiases[currentIndex]=0;//DELETE
                    currentIndex++;
                    for(int prevLayerIndex=0;prevLayerIndex<this.netStructure[layerIndex-1];prevLayerIndex++){//weigths leading into bias
                        this.allWeightsAndBiases[currentIndex]=getRandDoubleBetweenOneAndMinusOne(this.netStructure[layerIndex-1]);
                        currentIndex++;
                    }
                }
            }
        }
        this.feedThroughNet(new double[this.netStructure[0]]);
        double[]desiredOutput=new double[this.netStructure[this.netStructureLengthMinusOne]];
                //using 1/2 (act-des)^2
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allWeightsAndBiases.length];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        int totalAllExplored;
        int indexForAllGrad;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        int[] arrayIndexForAllWhenDoingWeightsAndError=new int[this.netStructure.length];
        int[]arrayForIndexForActivsError= new int[this.netStructure.length];
        int[]arrayForIndexForAllGrad=new int[this.netStructure.length];
        int []arrayForTotNeuronsExplored=new int[this.netStructure.length];
        for(int layerIndex=this.netStructure.length-1;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructure.length-1){
                activIndex=this.totalNeurons-this.netStructure[this.netStructure.length-1];
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]=(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
            }else{
                indexForAllWhenDoingWeightsAndError=0;
                for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }
                arrayIndexForAllWhenDoingWeightsAndError[layerIndex]=indexForAllWhenDoingWeightsAndError;
                indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }
                arrayForIndexForActivsError[layerIndex]=indexForActivsError;
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError]);
                    indexForActivsError++;
                }
            }
            //errorInLayer has been generated
            totalAllExplored=0;
            for(int layer = this.netStructure.length-1;layer>layerIndex-1;layer=layer-1){
                totalAllExplored=totalAllExplored + (this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
            }
            indexForAllGrad=this.allWeightsAndBiases.length-totalAllExplored;//first all of biases in this layer
            arrayForIndexForAllGrad[layerIndex]=indexForAllGrad;
            totNeuronsExplored=0;
            for(int layer=0;layer<layerIndex-1;layer++){//first index of activs in layer before
                totNeuronsExplored=totNeuronsExplored+this.netStructure[layer];
            }        
            arrayForTotNeuronsExplored[layerIndex]=totNeuronsExplored;
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
        this.indexForAllWeightAndError=arrayIndexForAllWhenDoingWeightsAndError;
        this.indexForActivError=arrayForIndexForActivsError;
        this.currentAll=arrayForIndexForAllGrad;
        this.totNeuron=arrayForTotNeuronsExplored;
    }
    public String getTestFileName(){
        return this.FILE_NAME_TEST;
    }
    public double[]loadBestNetWithThisArchitecture(){
        String lineRead;
        double []bestNet=new double[this.allWeightsAndBiases.length];
        double bestAcc=-0.5;
        int indexNo=0;
        int lineNo=0;
        int lineNoStart=0;
        int lineNoEnd=0;
        double currentAcc=0;
        boolean firstInstance=false;
        boolean isBest=true;
        boolean isSameArch=false;
        try{
            FileReader read = new FileReader(this.FILE_NAME);
            BufferedReader buff = new BufferedReader(read);
            Object[]a=buff.lines().toArray();
            String []s = new String[a.length];
            for(int i=0;i<a.length;i++){
                s[i]=String.valueOf(a[i]);
            }
            for(int i=0;i<s.length;i++){
                if(s[i].contains("Network Structure")){
                    isSameArch=false;
                    String str="Network Structure ";
                    for(int index=0;index<this.netStructure.length;index++){
                        str=str+String.valueOf(this.netStructure[index])+"  ";
                    }
                    if(s[i].equals(str)){
                        isSameArch=true;
                    }
                    //System.out.println("contains net struc but not found "+str+" d "+s[i]);
                }
                if(s[i].contains("Best network (with accuracy")&&isSameArch==true){
                    isBest=false;
                    String num="";
                    for(int c=0;c<s[i].length();c++){
                        boolean isNumOrDot=true;
                        try{
                            double f=Double.parseDouble(String.valueOf((s[i].charAt(c))));
                        }catch(NumberFormatException e){
                            if(s[i].charAt(c)!='.'){
                                isNumOrDot=false;
                            }
                        }
                        if(isNumOrDot==true){
                            num =num+s[i].charAt(c);
                        }
                    }
                    currentAcc=Double.parseDouble(num);
                    if(currentAcc>bestAcc&&isSameArch==true){
                        bestAcc=currentAcc;
                        isBest=true;
                    }
                    if(isBest==true&&isSameArch==true){
                        indexNo=0;
                        firstInstance=true;
                        lineNoStart=i+1;
                        lineNoEnd=i+bestNet.length;
                        int bestIn=0;
                        for(int element=lineNoStart;element<=lineNoEnd;element++){
                           // System.out.println("sdad "+s[element
                                   // ]+" elemnt no "+element+" tot "+this.allWeightsAndBiases.length);
                            bestNet[bestIn]=Double.parseDouble(s[element]);
                            bestIn++;
                        }
                    }
                }
            }
        }catch(IOException e){
            System.out.println("errro "+e+" with file "+FILE_NAME);
            e.printStackTrace();
        }
        return bestNet;
    }
    public static double getAnswerGivenArray(double[]out){
        double []choice={0,1,2,3,4,5,6,7,8,9};
        int highestIndex=0;
        for (int i = 0; i < out.length; i++) {
            if (i == 0) {
                highestIndex = 0;
            } else {
                if (out[i] > out[highestIndex]) {
                    highestIndex = i;
                }
            }
        }
        return choice[highestIndex];
    }


    public void shuffleArrayBarNoughtElement(String[]arr){
        int pos;
        String temp;
        Random rand = new Random();
        for(int i =1;i<arr.length;i++){
            do{
                pos = rand.nextInt(arr.length);
            }while(pos==0);
            temp = arr[i];
            arr[i]=arr[pos];
            arr[pos]=temp;
        }
    }
    public void shuffleTrainFile(){
        boolean error=true;
        String current[];
        do{
            try{
                FileReader reader = new FileReader(FILE_NAME_TRAIN);
                BufferedReader buff = new BufferedReader(reader);
                Stream<String> lines = buff.lines();
                Object[]a=lines.toArray();
                current = new String[a.length];
                for(int i=0;i<a.length;i++){
                    current[i]=(a[i].toString());
                    //System.out.println(current[i]);
                }
                lines.close();
                reader.close();
                buff.close();
                shuffleArrayBarNoughtElement(current);
                FileWriter  w =new FileWriter(FILE_NAME_TRAIN,false);
                BufferedWriter buffW=new BufferedWriter(w);
                for(int i =0;i<current.length;i++){
                    buffW.write(current[i]);
                    if(i!=current.length-1){
                        buffW.newLine();
                    }
                }
                w.flush();
                buffW.flush();
                w.close();
                buffW.close();
                error =false;
            }catch(IOException e){
                System.out.println("error "+e +" with file "+ FILE_NAME_TRAIN);
                e.printStackTrace();
            }
        }while(error==true);
    }
    public  double[][] getTrainingDataNew( int indexOfItemStart,int sampleSize){
        int maxIndex = indexOfItemStart+sampleSize-1;
        double[][]ret = new double[sampleSize][785];
        boolean error =true;
        do{
        int currentY=0;
        String[]line;
        try{
                FileReader reader = new FileReader(FILE_NAME_TRAIN);
                BufferedReader buff = new BufferedReader(reader);
                String lineRead;
                int lineNo=-2;
                while((lineRead=buff.readLine())!=null){
                    lineNo++;
                    if(lineNo>=indexOfItemStart&&lineNo<=maxIndex){
                        line=lineRead.split(",");
                        for(int i =0;i<line.length;i++){
                            ret[currentY][i]=Double.parseDouble(line[i]);
                        }
                        currentY++;
                    }
                    if(lineNo>maxIndex){
                        break;
                    }
                }
                buff.close();
                reader.close();
                error=false;
            }catch(IOException e ){
                System.out.println("error "+e +" with file "+ FILE_NAME_TRAIN);
                e.printStackTrace();
            }
        }while(error==true);
        return ret;
    }
    public void preComputeIndices(){
        double madeUpIn[]=new double[this.netStructure[0]];
        this.feedThroughNet(madeUpIn);
        double[]madeUpOut=new double[this.netStructure[this.netStructure.length-1]];
        calcGradientAttemptFirstRunForIndexes(madeUpOut);
    }
     public void calcGradientAttemptFirstRunForIndexes(double[]desiredOutput){
        //using 1/2 (act-des)^2
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allWeightsAndBiases.length];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        int totalAllExplored;
        int indexForAllGrad;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        int[] arrayIndexForAllWhenDoingWeightsAndError=new int[this.netStructure.length];
        int[]arrayForIndexForActivsError= new int[this.netStructure.length];
        int[]arrayForIndexForAllGrad=new int[this.netStructure.length];
        int []arrayForTotNeuronsExplored=new int[this.netStructure.length];
        for(int layerIndex=this.netStructure.length-1;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructure.length-1){
                activIndex=this.totalNeurons-this.netStructure[this.netStructure.length-1];
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]=(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
            }else{
                indexForAllWhenDoingWeightsAndError=0;
                for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }
                arrayIndexForAllWhenDoingWeightsAndError[layerIndex]=indexForAllWhenDoingWeightsAndError;
                indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }
                arrayForIndexForActivsError[layerIndex]=indexForActivsError;
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError]);
                    indexForActivsError++;
                }
            }
            //errorInLayer has been generated
            totalAllExplored=0;
            for(int layer = this.netStructure.length-1;layer>layerIndex-1;layer=layer-1){
                totalAllExplored=totalAllExplored + (this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
            }
            indexForAllGrad=this.allWeightsAndBiases.length-totalAllExplored;//first all of biases in this layer
            arrayForIndexForAllGrad[layerIndex]=indexForAllGrad;
            totNeuronsExplored=0;
            for(int layer=0;layer<layerIndex-1;layer++){//first index of activs in layer before
                totNeuronsExplored=totNeuronsExplored+this.netStructure[layer];
            }        
            arrayForTotNeuronsExplored[layerIndex]=totNeuronsExplored;
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
    }
     public static double mod(double x){
         if(x<0.0){
             x=x*-1;
         }
         return x;
     }
     public static double tanH(double x){
         return ((Math.pow(Math.E, x)-Math.pow(Math.E,-x))/(Math.pow(Math.E,x)+Math.pow(Math.E,-x)));
     }
     public static double gradTanH(double x){
         return(1.0-(Math.pow(tanH(x),2)));
     }
     public static double reLU(double x){
         if(x>0.0){
             return x;
         }else{
             return 0.0;
         }
     }
     public static double gradRELU(double x){
         if(x>0.0){
             return 1.0;
         }else{
             return 0.0;
        }
     }
     public static double[]softmax(double[]input){
         double sum=0.0;
         double[]ret=new double[input.length];
         for(int i =0;i<input.length;i++){
             sum =sum+Math.exp(input[i]);
         }
         for(int i =0;i<input.length;i++){
             ret[i]=Math.exp(input[i])/sum;
         }
         return ret;
     }
     
    public double[]calcGradientAttemptOptimised(double[]desiredOutput){
        //using  (act-des)^2 (sum of all in out layer)
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allLength];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        //int totalAllExplored;
        int indexForAllGrad;
        int changeD;
        //boolean allZero=false;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        for(int layerIndex=this.netStructureLengthMinusOne;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructureLengthMinusOne){
                activIndex=this.activIn;
                //allZero=false; //eroror is unlikeyl to be zero and hence not computationally worth checking for assume false, and propogate back to next layer
                //all error is extremely unlikely to be zero so not worthwile to check for cos softmax
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]= this.postSigActivs[activIndex] - desiredOutput[neuronInLayer];//softmax
                    //errorInLayer[neuronInLayer]=2*(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    //errorInLayer[neuronInLayer]=(-(desiredOutput[neuronInLayer]/this.postSigActivs[activIndex])+((1-desiredOutput[neuronInLayer])/(1-this.postSigActivs[activIndex])))*gr;//sigmoidDerivative(this.preSigActivs[activIndex])/*gradRELU(this.preSigActivs[activIndex])*/;
                    //errorInLayer[neuronInLayer]=((this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])/(mod(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])))*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
                
            }else{
                indexForAllWhenDoingWeightsAndError=this.indexForAllWeightAndError[layerIndex];//added in for opt
               /* for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }*/
                indexForActivsError=this.indexForActivError[layerIndex];
              /*  indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }*/
                changeD=1+this.netStructure[layerIndex];
                //allZero=true;
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        //indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                        indexWeightErrorChange=indexWeightErrorChange+changeD;
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError])/*gradRELU(this.preSigActivs[indexForActivsError])*/;
                    /*if(allZero){
                        if(errorInLayer[neuronInLayer]!=0.0){
                            allZero=false;//all zero is very unlikely
                        }
                    }*/
                    indexForActivsError++;
                }
            }
            indexForAllGrad=this.currentAll[layerIndex];
            totNeuronsExplored=this.totNeuron[layerIndex];
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                //grad[indexForAllGrad]=0;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
        return grad;
    }
    public void changeFiles(){
        try{
            FileReader readTest = new FileReader(FILE_NAME_TEST);
            BufferedReader buffTest = new BufferedReader(readTest);
            Object[]testOb=buffTest.lines().toArray();
            String newTest[]=new String[testOb.length+1];
            for(int i =0;i<newTest.length;i++){
                if(i==0){
                    newTest[i]="label,1x1,1x2,1x3,1x4,1x5,1x6,1x7,1x8,1x9,1x10,1x11,1x12,1x13,1x14,1x15,1x16,1x17,1x18,1x19,1x20,1x21,1x22,1x23,1x24,1x25,1x26,1x27,1x28,2x1,2x2,2x3,2x4,2x5,2x6,2x7,2x8,2x9,2x10,2x11,2x12,2x13,2x14,2x15,2x16,2x17,2x18,2x19,2x20,2x21,2x22,2x23,2x24,2x25,2x26,2x27,2x28,3x1,3x2,3x3,3x4,3x5,3x6,3x7,3x8,3x9,3x10,3x11,3x12,3x13,3x14,3x15,3x16,3x17,3x18,3x19,3x20,3x21,3x22,3x23,3x24,3x25,3x26,3x27,3x28,4x1,4x2,4x3,4x4,4x5,4x6,4x7,4x8,4x9,4x10,4x11,4x12,4x13,4x14,4x15,4x16,4x17,4x18,4x19,4x20,4x21,4x22,4x23,4x24,4x25,4x26,4x27,4x28,5x1,5x2,5x3,5x4,5x5,5x6,5x7,5x8,5x9,5x10,5x11,5x12,5x13,5x14,5x15,5x16,5x17,5x18,5x19,5x20,5x21,5x22,5x23,5x24,5x25,5x26,5x27,5x28,6x1,6x2,6x3,6x4,6x5,6x6,6x7,6x8,6x9,6x10,6x11,6x12,6x13,6x14,6x15,6x16,6x17,6x18,6x19,6x20,6x21,6x22,6x23,6x24,6x25,6x26,6x27,6x28,7x1,7x2,7x3,7x4,7x5,7x6,7x7,7x8,7x9,7x10,7x11,7x12,7x13,7x14,7x15,7x16,7x17,7x18,7x19,7x20,7x21,7x22,7x23,7x24,7x25,7x26,7x27,7x28,8x1,8x2,8x3,8x4,8x5,8x6,8x7,8x8,8x9,8x10,8x11,8x12,8x13,8x14,8x15,8x16,8x17,8x18,8x19,8x20,8x21,8x22,8x23,8x24,8x25,8x26,8x27,8x28,9x1,9x2,9x3,9x4,9x5,9x6,9x7,9x8,9x9,9x10,9x11,9x12,9x13,9x14,9x15,9x16,9x17,9x18,9x19,9x20,9x21,9x22,9x23,9x24,9x25,9x26,9x27,9x28,10x1,10x2,10x3,10x4,10x5,10x6,10x7,10x8,10x9,10x10,10x11,10x12,10x13,10x14,10x15,10x16,10x17,10x18,10x19,10x20,10x21,10x22,10x23,10x24,10x25,10x26,10x27,10x28,11x1,11x2,11x3,11x4,11x5,11x6,11x7,11x8,11x9,11x10,11x11,11x12,11x13,11x14,11x15,11x16,11x17,11x18,11x19,11x20,11x21,11x22,11x23,11x24,11x25,11x26,11x27,11x28,12x1,12x2,12x3,12x4,12x5,12x6,12x7,12x8,12x9,12x10,12x11,12x12,12x13,12x14,12x15,12x16,12x17,12x18,12x19,12x20,12x21,12x22,12x23,12x24,12x25,12x26,12x27,12x28,13x1,13x2,13x3,13x4,13x5,13x6,13x7,13x8,13x9,13x10,13x11,13x12,13x13,13x14,13x15,13x16,13x17,13x18,13x19,13x20,13x21,13x22,13x23,13x24,13x25,13x26,13x27,13x28,14x1,14x2,14x3,14x4,14x5,14x6,14x7,14x8,14x9,14x10,14x11,14x12,14x13,14x14,14x15,14x16,14x17,14x18,14x19,14x20,14x21,14x22,14x23,14x24,14x25,14x26,14x27,14x28,15x1,15x2,15x3,15x4,15x5,15x6,15x7,15x8,15x9,15x10,15x11,15x12,15x13,15x14,15x15,15x16,15x17,15x18,15x19,15x20,15x21,15x22,15x23,15x24,15x25,15x26,15x27,15x28,16x1,16x2,16x3,16x4,16x5,16x6,16x7,16x8,16x9,16x10,16x11,16x12,16x13,16x14,16x15,16x16,16x17,16x18,16x19,16x20,16x21,16x22,16x23,16x24,16x25,16x26,16x27,16x28,17x1,17x2,17x3,17x4,17x5,17x6,17x7,17x8,17x9,17x10,17x11,17x12,17x13,17x14,17x15,17x16,17x17,17x18,17x19,17x20,17x21,17x22,17x23,17x24,17x25,17x26,17x27,17x28,18x1,18x2,18x3,18x4,18x5,18x6,18x7,18x8,18x9,18x10,18x11,18x12,18x13,18x14,18x15,18x16,18x17,18x18,18x19,18x20,18x21,18x22,18x23,18x24,18x25,18x26,18x27,18x28,19x1,19x2,19x3,19x4,19x5,19x6,19x7,19x8,19x9,19x10,19x11,19x12,19x13,19x14,19x15,19x16,19x17,19x18,19x19,19x20,19x21,19x22,19x23,19x24,19x25,19x26,19x27,19x28,20x1,20x2,20x3,20x4,20x5,20x6,20x7,20x8,20x9,20x10,20x11,20x12,20x13,20x14,20x15,20x16,20x17,20x18,20x19,20x20,20x21,20x22,20x23,20x24,20x25,20x26,20x27,20x28,21x1,21x2,21x3,21x4,21x5,21x6,21x7,21x8,21x9,21x10,21x11,21x12,21x13,21x14,21x15,21x16,21x17,21x18,21x19,21x20,21x21,21x22,21x23,21x24,21x25,21x26,21x27,21x28,22x1,22x2,22x3,22x4,22x5,22x6,22x7,22x8,22x9,22x10,22x11,22x12,22x13,22x14,22x15,22x16,22x17,22x18,22x19,22x20,22x21,22x22,22x23,22x24,22x25,22x26,22x27,22x28,23x1,23x2,23x3,23x4,23x5,23x6,23x7,23x8,23x9,23x10,23x11,23x12,23x13,23x14,23x15,23x16,23x17,23x18,23x19,23x20,23x21,23x22,23x23,23x24,23x25,23x26,23x27,23x28,24x1,24x2,24x3,24x4,24x5,24x6,24x7,24x8,24x9,24x10,24x11,24x12,24x13,24x14,24x15,24x16,24x17,24x18,24x19,24x20,24x21,24x22,24x23,24x24,24x25,24x26,24x27,24x28,25x1,25x2,25x3,25x4,25x5,25x6,25x7,25x8,25x9,25x10,25x11,25x12,25x13,25x14,25x15,25x16,25x17,25x18,25x19,25x20,25x21,25x22,25x23,25x24,25x25,25x26,25x27,25x28,26x1,26x2,26x3,26x4,26x5,26x6,26x7,26x8,26x9,26x10,26x11,26x12,26x13,26x14,26x15,26x16,26x17,26x18,26x19,26x20,26x21,26x22,26x23,26x24,26x25,26x26,26x27,26x28,27x1,27x2,27x3,27x4,27x5,27x6,27x7,27x8,27x9,27x10,27x11,27x12,27x13,27x14,27x15,27x16,27x17,27x18,27x19,27x20,27x21,27x22,27x23,27x24,27x25,27x26,27x27,27x28,28x1,28x2,28x3,28x4,28x5,28x6,28x7,28x8,28x9,28x10,28x11,28x12,28x13,28x14,28x15,28x16,28x17,28x18,28x19,28x20,28x21,28x22,28x23,28x24,28x25,28x26,28x27,28x28";
                }else{
                    newTest[i]=testOb[i-1].toString();
                }
            }
            FileReader readTrain = new FileReader(FILE_NAME_TRAIN);
            BufferedReader buffTrain = new BufferedReader(readTrain);
            Object[]trainOb=buffTrain.lines().toArray();
            String newTrain[]=new String[trainOb.length+1];
            for(int i =0;i<newTrain.length;i++){
                if(i==0){
                    newTrain[i]="label,1x1,1x2,1x3,1x4,1x5,1x6,1x7,1x8,1x9,1x10,1x11,1x12,1x13,1x14,1x15,1x16,1x17,1x18,1x19,1x20,1x21,1x22,1x23,1x24,1x25,1x26,1x27,1x28,2x1,2x2,2x3,2x4,2x5,2x6,2x7,2x8,2x9,2x10,2x11,2x12,2x13,2x14,2x15,2x16,2x17,2x18,2x19,2x20,2x21,2x22,2x23,2x24,2x25,2x26,2x27,2x28,3x1,3x2,3x3,3x4,3x5,3x6,3x7,3x8,3x9,3x10,3x11,3x12,3x13,3x14,3x15,3x16,3x17,3x18,3x19,3x20,3x21,3x22,3x23,3x24,3x25,3x26,3x27,3x28,4x1,4x2,4x3,4x4,4x5,4x6,4x7,4x8,4x9,4x10,4x11,4x12,4x13,4x14,4x15,4x16,4x17,4x18,4x19,4x20,4x21,4x22,4x23,4x24,4x25,4x26,4x27,4x28,5x1,5x2,5x3,5x4,5x5,5x6,5x7,5x8,5x9,5x10,5x11,5x12,5x13,5x14,5x15,5x16,5x17,5x18,5x19,5x20,5x21,5x22,5x23,5x24,5x25,5x26,5x27,5x28,6x1,6x2,6x3,6x4,6x5,6x6,6x7,6x8,6x9,6x10,6x11,6x12,6x13,6x14,6x15,6x16,6x17,6x18,6x19,6x20,6x21,6x22,6x23,6x24,6x25,6x26,6x27,6x28,7x1,7x2,7x3,7x4,7x5,7x6,7x7,7x8,7x9,7x10,7x11,7x12,7x13,7x14,7x15,7x16,7x17,7x18,7x19,7x20,7x21,7x22,7x23,7x24,7x25,7x26,7x27,7x28,8x1,8x2,8x3,8x4,8x5,8x6,8x7,8x8,8x9,8x10,8x11,8x12,8x13,8x14,8x15,8x16,8x17,8x18,8x19,8x20,8x21,8x22,8x23,8x24,8x25,8x26,8x27,8x28,9x1,9x2,9x3,9x4,9x5,9x6,9x7,9x8,9x9,9x10,9x11,9x12,9x13,9x14,9x15,9x16,9x17,9x18,9x19,9x20,9x21,9x22,9x23,9x24,9x25,9x26,9x27,9x28,10x1,10x2,10x3,10x4,10x5,10x6,10x7,10x8,10x9,10x10,10x11,10x12,10x13,10x14,10x15,10x16,10x17,10x18,10x19,10x20,10x21,10x22,10x23,10x24,10x25,10x26,10x27,10x28,11x1,11x2,11x3,11x4,11x5,11x6,11x7,11x8,11x9,11x10,11x11,11x12,11x13,11x14,11x15,11x16,11x17,11x18,11x19,11x20,11x21,11x22,11x23,11x24,11x25,11x26,11x27,11x28,12x1,12x2,12x3,12x4,12x5,12x6,12x7,12x8,12x9,12x10,12x11,12x12,12x13,12x14,12x15,12x16,12x17,12x18,12x19,12x20,12x21,12x22,12x23,12x24,12x25,12x26,12x27,12x28,13x1,13x2,13x3,13x4,13x5,13x6,13x7,13x8,13x9,13x10,13x11,13x12,13x13,13x14,13x15,13x16,13x17,13x18,13x19,13x20,13x21,13x22,13x23,13x24,13x25,13x26,13x27,13x28,14x1,14x2,14x3,14x4,14x5,14x6,14x7,14x8,14x9,14x10,14x11,14x12,14x13,14x14,14x15,14x16,14x17,14x18,14x19,14x20,14x21,14x22,14x23,14x24,14x25,14x26,14x27,14x28,15x1,15x2,15x3,15x4,15x5,15x6,15x7,15x8,15x9,15x10,15x11,15x12,15x13,15x14,15x15,15x16,15x17,15x18,15x19,15x20,15x21,15x22,15x23,15x24,15x25,15x26,15x27,15x28,16x1,16x2,16x3,16x4,16x5,16x6,16x7,16x8,16x9,16x10,16x11,16x12,16x13,16x14,16x15,16x16,16x17,16x18,16x19,16x20,16x21,16x22,16x23,16x24,16x25,16x26,16x27,16x28,17x1,17x2,17x3,17x4,17x5,17x6,17x7,17x8,17x9,17x10,17x11,17x12,17x13,17x14,17x15,17x16,17x17,17x18,17x19,17x20,17x21,17x22,17x23,17x24,17x25,17x26,17x27,17x28,18x1,18x2,18x3,18x4,18x5,18x6,18x7,18x8,18x9,18x10,18x11,18x12,18x13,18x14,18x15,18x16,18x17,18x18,18x19,18x20,18x21,18x22,18x23,18x24,18x25,18x26,18x27,18x28,19x1,19x2,19x3,19x4,19x5,19x6,19x7,19x8,19x9,19x10,19x11,19x12,19x13,19x14,19x15,19x16,19x17,19x18,19x19,19x20,19x21,19x22,19x23,19x24,19x25,19x26,19x27,19x28,20x1,20x2,20x3,20x4,20x5,20x6,20x7,20x8,20x9,20x10,20x11,20x12,20x13,20x14,20x15,20x16,20x17,20x18,20x19,20x20,20x21,20x22,20x23,20x24,20x25,20x26,20x27,20x28,21x1,21x2,21x3,21x4,21x5,21x6,21x7,21x8,21x9,21x10,21x11,21x12,21x13,21x14,21x15,21x16,21x17,21x18,21x19,21x20,21x21,21x22,21x23,21x24,21x25,21x26,21x27,21x28,22x1,22x2,22x3,22x4,22x5,22x6,22x7,22x8,22x9,22x10,22x11,22x12,22x13,22x14,22x15,22x16,22x17,22x18,22x19,22x20,22x21,22x22,22x23,22x24,22x25,22x26,22x27,22x28,23x1,23x2,23x3,23x4,23x5,23x6,23x7,23x8,23x9,23x10,23x11,23x12,23x13,23x14,23x15,23x16,23x17,23x18,23x19,23x20,23x21,23x22,23x23,23x24,23x25,23x26,23x27,23x28,24x1,24x2,24x3,24x4,24x5,24x6,24x7,24x8,24x9,24x10,24x11,24x12,24x13,24x14,24x15,24x16,24x17,24x18,24x19,24x20,24x21,24x22,24x23,24x24,24x25,24x26,24x27,24x28,25x1,25x2,25x3,25x4,25x5,25x6,25x7,25x8,25x9,25x10,25x11,25x12,25x13,25x14,25x15,25x16,25x17,25x18,25x19,25x20,25x21,25x22,25x23,25x24,25x25,25x26,25x27,25x28,26x1,26x2,26x3,26x4,26x5,26x6,26x7,26x8,26x9,26x10,26x11,26x12,26x13,26x14,26x15,26x16,26x17,26x18,26x19,26x20,26x21,26x22,26x23,26x24,26x25,26x26,26x27,26x28,27x1,27x2,27x3,27x4,27x5,27x6,27x7,27x8,27x9,27x10,27x11,27x12,27x13,27x14,27x15,27x16,27x17,27x18,27x19,27x20,27x21,27x22,27x23,27x24,27x25,27x26,27x27,27x28,28x1,28x2,28x3,28x4,28x5,28x6,28x7,28x8,28x9,28x10,28x11,28x12,28x13,28x14,28x15,28x16,28x17,28x18,28x19,28x20,28x21,28x22,28x23,28x24,28x25,28x26,28x27,28x28";
                }else{
                    newTrain[i]=trainOb[i-1].toString();
                }
            }
            readTest.close();
            buffTest.close();
            readTrain.close();
            buffTrain.close();
            FileWriter writeTest = new FileWriter(FILE_NAME_TEST,false);
            BufferedWriter buffWriteTest = new BufferedWriter(writeTest);
            for(int i =0;i<newTest.length;i++){
                buffWriteTest.write(newTest[i]);
                if(i!=newTest.length-1){
                    buffWriteTest.newLine();
                }
            }
            FileWriter writeTrain = new FileWriter(FILE_NAME_TRAIN,false);
            BufferedWriter buffWriteTrain = new BufferedWriter(writeTrain);
            for(int i =0;i<newTrain.length;i++){
                buffWriteTrain.write(newTrain[i]);
                if(i!=newTrain.length-1){
                    buffWriteTrain.newLine();
                }
            }
            writeTest.flush();
            buffWriteTest.flush();
            writeTest.close();
            buffWriteTest.close();
            writeTrain.flush();
            buffWriteTrain.flush();
            writeTrain.close();
            buffWriteTrain.close();
        }catch(IOException e){
            System.out.println("E "+e);
            e.printStackTrace();
        }
    }
    public double[][] loadTrainingDataFromFile (int totalTrainSize,int across){
       boolean error=true;
       double[][]ret=new double[totalTrainSize][across];
       do{
        try{
            FileReader read = new FileReader(FILE_NAME_TRAIN);
            BufferedReader buff = new BufferedReader(read);
            Stream<String>l= buff.lines();
            Object a[]=l.toArray();
            l.close();
            buff.close();
            read.close();
            String line;
            String lineRead[];
            for(int y=0;y<a.length;y++){
                if(y!=0){
                    line = String.valueOf(a[y]);
                    lineRead=line.split(",");
                    for(int x =0;x<lineRead.length;x++){
                        ret[y-1][x]=Double.parseDouble(lineRead[x]);
                    }
                }
                
            }
            error=false;
        }catch(IOException e){
            System.out.println("Error with "+e+" for file "+FILE_NAME_TRAIN);
            e.printStackTrace();
        }
       }while(error==true);
        return ret;
    } 
    public void betterTestAndTrain(){
        try{
            System.out.println("startong ads");
            FileReader readTrain=new FileReader(FILE_NAME_TRAIN);
            BufferedReader buffTrain = new BufferedReader(readTrain);
            Object []trainOb=buffTrain.lines().toArray();
            String[]train=new String[trainOb.length-1];
            for(int i=0;i<trainOb.length;i++){
                if(i!=0){
                    train[i-1]=trainOb[i].toString();
                }
            }
            FileReader readTest=new FileReader(FILE_NAME_TEST);
            BufferedReader buffTest = new BufferedReader(readTest);
            Object []testOb=buffTest.lines().toArray();
            String[]test=new String[testOb.length-1];
            for(int i=0;i<testOb.length;i++){
                if(i!=0){
                    test[i-1]=testOb[i].toString();
                    
                }
            }

            String all[]=new String[test.length+train.length];
            for(int i=0;i<test.length;i++){
                all[i]=test[i];
            }
            for(int i =test.length;i<test.length+train.length;i++){
                all[i]=train[i-test.length];
            }
                    int pos;
            String temp;
            System.out.println("pre shuffle");
            Random rand = new Random();
            for(int i =0;i<all.length;i++){
                pos = rand.nextInt(all.length);
                temp = all[i];
                all[i]=all[pos];
                all[pos]=temp;
            }
            System.out.println("post shuffle");
            int currentDigitNewTest=0;
            int currentDigitNewTrain=0;
            String[]newTest=new String[10000];
            String []newTrain=new String[60000];
            int noOfDigit;
            for(int digit=0;digit<10;digit++){
                System.out.println("doing digit "+digit);
                noOfDigit=0;
                for(int i=0;i<all.length;i++){
                    if(all[i].split(",")[0].equals(String.valueOf(digit))){
                        if(noOfDigit<6000){
                            newTrain[currentDigitNewTrain]=all[i];
                            currentDigitNewTrain++;
                        }else{
                            newTest[currentDigitNewTest]=all[i];
                            currentDigitNewTest++;
                        }
                        noOfDigit++;
                    }
                }
                System.out.println("no of digits done "+(noOfDigit));
            }
            FileWriter writeNewTest = new FileWriter("newTest.csv",false);
            BufferedWriter buffNewTest = new BufferedWriter(writeNewTest);
            buffNewTest.write(testOb[0].toString());
            for(int i =0;i<newTest.length;i++){
                buffNewTest.newLine();
                buffNewTest.write(newTest[i]);
            }
            buffNewTest.flush();
            writeNewTest.flush();
            buffNewTest.close();
            writeNewTest.close();
            FileWriter writeNewTrain = new FileWriter("newTrain.csv",false);
            BufferedWriter buffNewTrain = new BufferedWriter(writeNewTrain);
            buffNewTrain.write(trainOb[0].toString());
            for(int i =0;i<newTrain.length;i++){
                buffNewTrain.newLine();
                buffNewTrain.write(newTrain[i]);
            }
            buffNewTrain.flush();
            writeNewTrain.flush();
            buffNewTrain.close();
            writeNewTrain.close();
            buffTrain.close();
            readTrain.close();
            buffTest.close();
            readTest.close();
        }catch(IOException e){
            System.out.println("Error "+e );
            e.printStackTrace();;
        }
    }
    public double[][]getTrainingDataFromMemory(double[][]trainingData,int indexOfTrain,int sampleSize){
        int lastInex=indexOfTrain+sampleSize;
        double[][]dat =new double[sampleSize][785];
        int yIn=0;
        for(int y=indexOfTrain;y<lastInex;y++){
            for(int x =0;x<trainingData[0].length;x++){
                dat[yIn][x]=trainingData[y][x];
            }
            yIn++;
        }
        return dat;
    }
    public void saveDeatils(int noOfEpochs, double accuracy, double timeTaken, double learningRate,double highestAcc,double[]bestNet,int sampleSize){
         try{
            FileWriter writer = new FileWriter(FILE_NAME,true);
            BufferedWriter bufferedWriter = new BufferedWriter(writer);
            bufferedWriter.write("----------------------------");
            bufferedWriter.newLine();
            bufferedWriter.write("Network Structure ");
            for(int i =0;i<this.getNetworkStructure().length;i++){
                bufferedWriter.write(String.valueOf(this.getNetworkStructure()[i])+"  ");
            }
            bufferedWriter.newLine();
            bufferedWriter.write("All weights and biases");
            bufferedWriter.newLine();
            for(int i =0;i<this.getAllWeigthsAndBiases().length;i++){
                bufferedWriter.write(String.valueOf(this.getAllWeigthsAndBiases()[i]));
                bufferedWriter.newLine();
            }
            bufferedWriter.write("No of epochs "+noOfEpochs);
            bufferedWriter.newLine();
            bufferedWriter.write("Final accuracy of "+accuracy);
            bufferedWriter.newLine();
            bufferedWriter.write("time taken in milliseconds "+timeTaken);
            bufferedWriter.newLine();
            bufferedWriter.write("Learning rate of "+learningRate);
            bufferedWriter.newLine();
            bufferedWriter.write("Best network (with accuracy "+highestAcc+"% with this sturcture had a configuration of");
            bufferedWriter.newLine();
            for(int i=0;i<bestNet.length;i++){
                bufferedWriter.write(String.valueOf(bestNet[i]));
                bufferedWriter.newLine();
            }
            bufferedWriter.newLine();
            bufferedWriter.write("Using a sample size of "+sampleSize+" for each epoch out of a total of 60000 samples. Note this doesnt neccesarly apply if this is true:  false");
            bufferedWriter.newLine();
            bufferedWriter.write("----------------------------");
            bufferedWriter.close();
            writer.close();
        }catch(IOException e){
            System.out.println("error with file "+FILE_NAME+" error of "+e);
            System.out.println("Error trace");
            e.printStackTrace();
        }
    }
     public void trainNew(double learningRate, int noOfEpochs,int sampleSize,int totalTrainSize,int printAndTestAfterThisManyIterations,boolean loadTrainInMemory){
        double changeBy[];
        long timeBefore=System.currentTimeMillis();
        double[][]data;//each across is one full set of data index 0 is label
        //double out[];
        final int epoc=noOfEpochs;
        final int samp=sampleSize;
        final int h=(totalTrainSize/sampleSize);
        final double sampD=Double.parseDouble(String.valueOf(samp));
        final double rate=learningRate;
        final boolean loadInMem=loadTrainInMemory;
        final int trainS=totalTrainSize;
        final int printAfter=printAndTestAfterThisManyIterations;
        double[]bestNet = new double[this.allLength];
        double accuracy;
        double mT[]=new double[this.allLength];
        double vT[]=new double[this.allLength];
        double mHatT[]=new double[this.allLength];
        double vHatT[]=new double[this.allLength];
        double prevMT[]=new double[this.allLength];
        double prevVT[]=new double[this.allLength];
        double t =0.0;
        try{
            double l =this.previousVHat[0];
            for(int i =0;i<this.previousMHat.length;i++){
                prevMT[i]=this.previousMHat[i];
                prevVT[i]=this.previousVHat[i];    
            }
            t=this.timeStep;
        }catch(java.lang.NullPointerException e){
        }
        final double betaOne=0.9;
        final double betaTwo=0.999;
        final double epsilon = Math.pow(10,-8);
        

        double gradOfOne[];
        int indexOfTrain=0;
        double[][]trainingData=new double[1][1];
        if(loadInMem==true){
            trainingData=new double[trainS][785];
        }
        int indexPrintAndTest=1;
        double start[]=this.testAccuracy();
        double highestAcc=start[0];
        for(int i =0;i<this.allLength;i++){
            bestNet[i]=this.allWeightsAndBiases[i];
        }
        double desiredOut[];
        double[]inputToNet=new double[this.netStructure[0]];
        double correctAnswer;
        System.out.println("Accuracy before training "+highestAcc+" %"+" loss of "+start[1]);
        for(int epoch=1;epoch<=epoc;epoch++){
            shuffleTrainFile();
            if(loadInMem==true){
                trainingData=loadTrainingDataFromFile(trainS,785);
            }
            indexOfTrain=0;
            for(int iteration=1;iteration<=h;iteration++){
                changeBy = new double[this.allLength];
                if(loadInMem==false){
                    data = getTrainingDataNew(indexOfTrain,samp);
                }else{
                    data = getTrainingDataFromMemory(trainingData,indexOfTrain,samp);
                }
                indexOfTrain=indexOfTrain+samp;
                for (int sampleNo = 0; sampleNo < samp; sampleNo++) {
                    correctAnswer = data[sampleNo][0];
                    for (int i = 0; i < this.netStructure[0]; i++) {
                        inputToNet[i] = (data[sampleNo][i + 1]/(255.0));
                    }
                    this.feedThroughNet(inputToNet);
                    desiredOut = getDesiredOut(correctAnswer);
                    gradOfOne = this.calcGradientAttemptOptimised(desiredOut);
                    for (int i = 0; i < this.allWeightsAndBiases.length; i++) {
                        //changeBy[i] = Double.parseDouble(String.valueOf((changeBy[i] * sampleNo + gradOfOne[i]))) / Double.parseDouble(String.valueOf(sampleNo + 1.0));//saves massives memory but sliggtly slower
                        changeBy[i]=changeBy[i]+gradOfOne[i];
                    }
                }
                t=t+1.0;
                for(int i=0;i<changeBy.length;i++){
                    changeBy[i]=changeBy[i]/sampD;
                    mT[i]=betaOne*prevMT[i]+(1.0-betaOne) * changeBy[i];
                    vT[i]=betaTwo*prevVT[i]+(1.0-betaTwo)*(changeBy[i]*changeBy[i]);
                    mHatT[i]=mT[i]/(1.0-Math.pow(betaOne,t));
                    vHatT[i]=vT[i]/(1.0-Math.pow(betaTwo,t));
                    prevMT[i]=mT[i];
                    prevVT[i]=vT[i];
                    this.allWeightsAndBiases[i]=this.allWeightsAndBiases[i] - (rate*mHatT[i]/(Math.sqrt(vHatT[i])+epsilon));
                    
                    //this.allWeightsAndBiases[i]=this.allWeightsAndBiases[i]-(changeBy[i]*rate);;
                    //System.out.println("mt "+mT[i]+" vt "+vT[i]+" mht "+mHatT[i]+" vHat "+vHatT[i]+" grad "+changeBy[i]+" changeBy "+var+" value before change "+this.allWeightsAndBiases[i]);
                    
                }
                if(indexPrintAndTest==printAfter){
                    double[]accAndLoss=this.testAccuracy();
                    accuracy = accAndLoss[0];
                    if (accuracy >= highestAcc || highestAcc <= 0) {
                        for (int i = 0; i < this.allWeightsAndBiases.length; i++) {
                            bestNet[i] = this.allWeightsAndBiases[i];
                        }
                        highestAcc = accuracy;
                        //saveDeatils(noOfEpochs, accuracy, 0, learningRate, highestAcc, bestNet, sampleSize);
                    }
                    System.out.println("Accuracy " + accuracy + "% after " + epoch + " epochs out of " + noOfEpochs + " epochs. Sample size of " + sampleSize + " with a learning rate of " + rate+" loss of "+accAndLoss[1]+ "iteration "+(iteration)+" out of "+h+" best accuracy of "+highestAcc);
                    indexPrintAndTest=0;
                }
                indexPrintAndTest++;
            }
        }
        this.timeStep=t;
        this.previousMHat=new double[this.allLength];
        this.previousVHat=new double[this.allLength];
        for(int i =0;i<prevMT.length;i++){
            this.previousMHat[i]=prevMT[i];
            this.previousVHat[i]=prevVT[i];
        }
        accuracy = this.testAccuracy()[0];
        long timeAtEnd = System.currentTimeMillis();
        long timeTaken=timeAtEnd-timeBefore;
        try{
            FileWriter writer = new FileWriter(FILE_NAME,false);
            BufferedWriter bufferedWriter = new BufferedWriter(writer);
            bufferedWriter.write("----------------------------");
            bufferedWriter.newLine();
            bufferedWriter.write("Network Structure ");
            for(int i =0;i<this.getNetworkStructure().length;i++){
                bufferedWriter.write(String.valueOf(this.getNetworkStructure()[i])+"  ");
            }
            bufferedWriter.newLine();
            bufferedWriter.write("All weights and biases");
            bufferedWriter.newLine();
            for(int i =0;i<this.getAllWeigthsAndBiases().length;i++){
                bufferedWriter.write(String.valueOf(this.getAllWeigthsAndBiases()[i]));
                bufferedWriter.newLine();
            }
            bufferedWriter.write("No of epochs "+noOfEpochs);
            bufferedWriter.newLine();
            bufferedWriter.write("Final accuracy of "+accuracy);
            bufferedWriter.newLine();
            bufferedWriter.write("time taken in milliseconds "+timeTaken);
            bufferedWriter.newLine();
            bufferedWriter.write("Learning rate of "+learningRate);
            bufferedWriter.newLine();
            bufferedWriter.write("Best network (with accuracy "+highestAcc+"% with this sturcture had a configuration of");
            bufferedWriter.newLine();
            for(int i=0;i<bestNet.length;i++){
                bufferedWriter.write(String.valueOf(bestNet[i]));
                bufferedWriter.newLine();
            }
            bufferedWriter.newLine();
            bufferedWriter.write("Using a sample size of "+sampleSize+" for each epoch out of a total of 60000 samples. Note this doesnt neccesarly apply if this is true:  false");
            bufferedWriter.newLine();
            bufferedWriter.write("----------------------------");
            bufferedWriter.close();
            writer.close();
        }catch(IOException e){
            System.out.println("error with file "+FILE_NAME+" error of "+e);
            System.out.println("Error trace");
            e.printStackTrace();
        }
    }
    public double[][]getTrainingData(int sampleSize){
        double out[][]=new double[sampleSize][785];
        boolean error=true;
        do{
            int sizeOfData=60000;//line 1 is unusable and line 60001 is last line
            Random rnd = new Random();
            int startingLine=rnd.nextInt((sizeOfData+1)-sampleSize)+2;
            int lineNo=1;
            String lineRead;

            int currentY =0;
            String []line;
            try{
                FileReader reader = new FileReader(FILE_NAME_TRAIN);
                BufferedReader buff = new BufferedReader(reader);
                while((lineRead=buff.readLine())!=null){
                    if(lineNo>=startingLine && lineNo<startingLine+sampleSize){
                        line = lineRead.split(",");
                        for(int i =0;i<line.length;i++){
                            if(i==0){
                                out[currentY][i]=Double.parseDouble(line[i]);
                            }else{
                                out[currentY][i]=Double.parseDouble(line[i])/255.0;
                            }
                        }
                        currentY++;
                    }else if(lineNo>startingLine+sampleSize){
                        break;
                    }
                    lineNo++;

                }
                buff.close();
                reader.close();
                error=false;
            }catch(IOException e){
               System.out.println("Error "+e+" With file "+FILE_NAME_TRAIN);
               e.printStackTrace();
            }
        }while(error==true);
        return out;
    }
    public double[]getDesiredOut(double correctAnswer){
        double outs[]={0,1,2,3,4,5,6,7,8,9};
        double desired[]=new double[outs.length];
        for(int i=0;i<desired.length;i++){
            if(outs[i]==correctAnswer){
                desired[i]=1;
            }else{
                desired[i]=0;
            }
        }
        return desired;
    }
    public NeuralNetwork copyNetwork(){
        NeuralNetwork network = new NeuralNetwork(this.netStructure);
        network.setAllWeightsAndBiases(this.allWeightsAndBiases);
        return network;
    }
    public double[] testAccuracy(){
        int totalCorrect=0;
        int totalIncorrect=0;
        String lineRead;
        int totalSample=0;
        String line[];
        double averageLoss=0;
        double[]in= new double[784];
        double out[];
        int correctAnswer;
        double[] chosenAnswer;
        int highestIndex=0;
        double lossThisInstance;
        double []choice={0,1,2,3,4,5,6,7,8,9};
        int lineNo=1;
        try{
            FileReader reader = new FileReader(FILE_NAME_TEST);
            BufferedReader buff = new BufferedReader(reader);
            while((lineRead=buff.readLine())!=null){
                if(lineNo!=1){
                    line = lineRead.split(",");
                    correctAnswer = Integer.parseInt(line[0]);
                    for(int i =1; i<line.length;i++){
                        in[i-1]=(Double.parseDouble(line[i])/(255.0));
                    }
                    out = this.feedThroughNet(in);
                    chosenAnswer=getDesiredOut(correctAnswer);
                    lossThisInstance=0;
                    for(int i =0;i<out.length;i++){
                        
                        lossThisInstance=lossThisInstance+(chosenAnswer[i]*Math.log(out[i]));
                        //lossThisInstance=lossThisInstance+((out[i]-chosenAnswer[i])*(out[i]-chosenAnswer[i]));
                        if(i==0){
                            highestIndex=0;
                        }
                        else{
                            if(out[i]>out[highestIndex]){
                                highestIndex=i;
                            }
                        }
                    }
                    //lossThisInstance=lossThisInstance/(double)out.length;
                    lossThisInstance=lossThisInstance*-1.0;
                    averageLoss=averageLoss+lossThisInstance;
                    if(choice[highestIndex]==correctAnswer){
                        totalCorrect++;
                    }else{
                        totalIncorrect++;
                    }
                    totalSample++;
                }
                lineNo++;
            }
            lineNo=lineNo-2;
            averageLoss=averageLoss/(double)lineNo;
        }catch(IOException e){
            System.out.println("Error "+e +" with file "+FILE_NAME_TEST);
            e.printStackTrace();
            return this.testAccuracy();
            
        }
        double re = (double)totalCorrect/(double)totalSample;
        re = re *100.0;
        double ret[]={re,averageLoss};
        return ret;
    }
 
    public double[]loadSettingFromFile(double finalAccuracy){
        String lineRead;
        double[]re=new double[this.getAllWeigthsAndBiases().length];
        int indexNo=0;
        int lineNo=0;
        int lineNoStart=0;
        int lineNoEnd=0;
        boolean firstInstance=false;
        try{
            FileReader read = new FileReader(this.FILE_NAME);
            BufferedReader buff = new BufferedReader(read);
            while((lineRead=buff.readLine())!=null){
                if(lineRead.equals("All weights and biases")){
                    indexNo=0;
                    firstInstance=true;
                    lineNoStart=lineNo+1;
                    lineNoEnd=lineNo+re.length;
                }
                if(firstInstance==true&&lineNo>=lineNoStart&&lineNo<=lineNoEnd){
                    re[indexNo]=Double.parseDouble(lineRead);
                    indexNo++;
                }
                lineNo++;
                if(lineRead.equals("Final accuracy of "+finalAccuracy)){
                    break;
                }
            }
        }catch(IOException e){
            System.out.println("errro "+e);
        }
        return re;
    }
    public int[]getNetworkStructure(){
        return this.netStructure;
    }
    public double[] getAllWeigthsAndBiases(){
        return this.allWeightsAndBiases;
    }
    public double[]feedThroughNet(double[]input){
        int currentAllIndex=0;
        double []activsThisLayer;
        double []out = new double[this.netStructure[this.netStructureLengthMinusOne]];
        int indexForActivsInThisLayer;
        double valueOfBias;
        double activVal;
        double[]activsPrevLayer=input;
        int indexForPreSigActivs=0;
        int indexForPostSigActivs=0;
        this.preSigActivs=new double[this.totalNeurons];
        this.postSigActivs=new double[this.totalNeurons];
        for(int i=0;i<this.netStructure[0];i++){//first layer
            this.preSigActivs[indexForPreSigActivs]=input[i];
            this.postSigActivs[indexForPostSigActivs]=input[i];
            indexForPostSigActivs++;
            indexForPreSigActivs++;
        }
        for(int layerIndex=1;layerIndex<this.netStructureLength;layerIndex++){
            indexForActivsInThisLayer=0;
            activsThisLayer=new double[this.netStructure[layerIndex]];
            for(int indexOfLayer=0;indexOfLayer<this.netStructure[layerIndex];indexOfLayer++){//neurons in current kayer
                valueOfBias=this.allWeightsAndBiases[currentAllIndex];
                currentAllIndex++;
                activVal=0;
                for(int prevLayerIndex=0;prevLayerIndex<this.netStructure[layerIndex-1];prevLayerIndex++){//neurons in orev layer
                    activVal=activVal + activsPrevLayer[prevLayerIndex] * this.allWeightsAndBiases[currentAllIndex];
                    currentAllIndex++;

                }
                this.preSigActivs[indexForPreSigActivs]=activVal+valueOfBias;
                indexForPreSigActivs++;
                if(layerIndex!=this.netStructureLengthMinusOne){
                    activsThisLayer[indexForActivsInThisLayer]=sigmoid(this.preSigActivs[indexForPreSigActivs-1]);/*activsThisLayer[indexForActivsInThisLayer]=reLU(this.preSigActivs[indexForPreSigActivs-1]);*/
                    indexForActivsInThisLayer++;
                    this.postSigActivs[indexForPostSigActivs]=activsThisLayer[indexForActivsInThisLayer-1];
                    indexForPostSigActivs++;
                }else if(indexOfLayer==this.netStructure[layerIndex]-1){
                    double preSigLayer[]=new double[this.netStructure[layerIndex]];
                    for(int indexVal=0;indexVal<preSigLayer.length;indexVal++){
                       preSigLayer[indexVal]=this.preSigActivs[this.totalNeuronsMinusLastLayer+indexVal];
                    }
                    activsThisLayer=softmax(preSigLayer);
                    for(int i=0;i<activsThisLayer.length;i++){
                        this.postSigActivs[indexForPostSigActivs]=activsThisLayer[i];
                        indexForPostSigActivs++;
                    }
                }
            }
            activsPrevLayer=activsThisLayer;
            if(layerIndex==this.netStructureLengthMinusOne){
                out = activsThisLayer;
            }
        }

        return out;
    }
    
   

    public double[]calcGradientAttemptFour(double[]desiredOutput){
        //using 1/2 (act-des)^2
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allWeightsAndBiases.length];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        int totalAllExplored;
        int indexForAllGrad;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        for(int layerIndex=this.netStructure.length-1;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructure.length-1){
                activIndex=this.totalNeurons-this.netStructure[layerIndex];
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]=(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
            }else{
                indexForAllWhenDoingWeightsAndError=0;
                for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }
                indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError]);
                    indexForActivsError++;
                }
            }
            //errorInLayer has been generated
            totalAllExplored=0;
            for(int layer = this.netStructure.length-1;layer>layerIndex-1;layer=layer-1){
                totalAllExplored=totalAllExplored + (this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
            }
            indexForAllGrad=this.allWeightsAndBiases.length-totalAllExplored;//first all of biases in this layer
            totNeuronsExplored=0;
            for(int layer=0;layer<layerIndex-1;layer++){//first index of activs in layer before
                totNeuronsExplored=totNeuronsExplored+this.netStructure[layer];
            }        
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
        //for(int i =0;i<grad.length;i++){
            //System.out.println("grad["+i+"] = "+grad[i]);
        //}
        return grad;
    }
    
    
  
    public void printPostSigActvis(){
        for(double val:this.postSigActivs){
            System.out.println(val);
        }
    }
    public void printPreSigActivs(){
        for(double val:this.preSigActivs){
            System.out.println(val);
        }
    }
    public void printAllWeightsAndBiases(){
        for(int i =0;i<this.allWeightsAndBiases.length;i++){
            System.out.println(this.allWeightsAndBiases[i]);
        }
    }
    public void setAllWeightsAndBiases(double[]allWAndB){
        if(allWAndB.length!=this.allWeightsAndBiases.length){
            System.out.println("Attempted to change weights and biases to an array of different length ");
        }
        for(int i =0;i<allWAndB.length;i++){
            this.allWeightsAndBiases[i]=allWAndB[i];
        }
        //this.allWeightsAndBiases=allWAndB;
    }
    public static double leakyReLU(double x){
        if(x>0.0){
            return x;
        }else{
            return 0.01;
        }
    }
    public static double leakyReLUGrad(double x){
        if(x>0.0){
            return 1;
        }else{
            return 0.01;
        }
    }
    public static double sigmoid(double x){
        if(x>0.0){
            return x;
        }else{
            return 0.0;
        }
    }
    public static double sigmoidDerivative(double x){
       /* if(x==0.0){
            System.out.println("zero deriv error");
        }*/
       if(x>0.0){
            return 1;
        }else{
            return 0;
        }

    }
    public static double getRandDoubleBetweenOneAndMinusOne(int numberOfInputsForWeightsInThisLayer){
        
        Random rnd = new Random();
        return (rnd.nextGaussian()*Math.sqrt(2.0/(double)numberOfInputsForWeightsInThisLayer));
    }
}
