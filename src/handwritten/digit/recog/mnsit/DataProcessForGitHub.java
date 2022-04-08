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

/**
 *
 * @author mortimer
 */
public class DataProcessForGitHub {
        public static void processDataForGitHubFileSizeLimit(){
        try{
            FileReader read = new FileReader("dataProcessedForGitHubSizeLimit.txt");
            BufferedReader buffRead = new BufferedReader(read);
            if(buffRead.readLine().equals("true")){
                buffRead.close();read.close();
                return;
            }
            buffRead.close();read.close();
            String train[] =new String[60001];
            FileReader one = new FileReader("one.txt");
            BufferedReader buffOne = new BufferedReader(one);
            for(int i=0;i<30000;i++){
                train[i]=buffOne.readLine();
            }buffOne.close();one.close();
            FileReader two = new FileReader("two.txt");
            BufferedReader buffTwo = new BufferedReader(two);
            for(int i=30000;i<60001;i++){
                train[i]=buffTwo.readLine();
            }buffTwo.close();two.close();
            FileWriter write = new FileWriter("mnist_train.csv",false);
            BufferedWriter buffWrite = new BufferedWriter(write);
            buffWrite.write(train[0]);
            for(int i=1;i<train.length;i++){
                buffWrite.newLine();
                buffWrite.write(train[i]);
            }buffWrite.flush();write.flush();
            buffWrite.close();write.close();
            FileWriter w = new FileWriter("dataProcessedForGitHubSizeLimit.txt",false);
            BufferedWriter buffW = new BufferedWriter(w);
            buffW.write("true");
            buffW.flush();w.flush();buffW.close();w.close();
        }catch(IOException e){
            
        }
    }
    public static void splitIntoTwoFilesGitHubLimit(){
        String train[]=new String[60001];
        try{
            FileReader read = new FileReader("mnist_train.csv");
            BufferedReader buffRead = new BufferedReader(read);
            for(int in=0;in<train.length;in++){
                train[in]=buffRead.readLine();
            }buffRead.close();read.close();
            FileWriter one = new FileWriter("one.txt",false);
            BufferedWriter buffOne = new BufferedWriter(one);
            buffOne.write(train[0]);
            for(int i=1;i<30000;i++){
                buffOne.newLine();
                buffOne.write(train[i]);
            }buffOne.flush();one.flush();
            buffOne.close();one.close();
            FileWriter two = new FileWriter("two.txt",false);
            BufferedWriter buffTwo = new BufferedWriter(two);
            buffTwo.write(train[30000]);
            for(int i=30001;i<60001;i++){
                buffTwo.newLine();
                buffTwo.write(train[i]);
            }buffTwo.flush();two.flush();
            buffTwo.close();two.close();
        }catch(IOException e){
            
        }
    }
}
