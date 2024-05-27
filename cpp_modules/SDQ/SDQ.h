// load.h

// MIT License

// Copyright (c) 2022 deponce(Linfeng Ye), Ahmed Hussein Salamah, Kaixiang Zheng, University of Waterloo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "SDQ_Class.h"

const float EPS = 0.001;

void SDQ::opt_RS_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    int i;
    // C channel
    std::fill_n(SDQ::Block.state.ID, 64,0);
    std::fill_n(SDQ::Block.state.rs, 64,0);
    SDQ::Block.ent.clear();
    SDQ::Block.P.clear();
    // initialize Py0
    
    
    for(i=0; i<SDQ::seq_len_Y; i++){
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], SDQ::RSlst, SDQ::IDlst);
        cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.P);
    }
    
    SDQ::RSlst.clear(); SDQ::IDlst.clear();
    norm(SDQ::Block.P, SDQ::Block.ent);
    cal_ent(SDQ::Block.ent);
    SDQ::Block.set_channel('S');
    SDQ::Block.set_Q_table(SDQ::Q_table_Y); 
    SDQ::Loss = 0;
    SDQ::Block.state.ent=0;
    
    
    for(i=0; i<SDQ::seq_len_Y; i++){
        std::fill_n(SDQ::Block.state.ID, 64,0);
        std::fill_n(SDQ::Block.state.rs, 64,0);
        SDQ::Block.cal_RS(seq_dct_coefs_Y[i],
                          seq_dct_idxs_Y[i],
                          SDQ::RSlst, SDQ::IDlst);
        RSlst_to_Block(seq_dct_idxs_Y[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Y[i]);
        SDQ::RSlst.clear();SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;
    }
}


void SDQ::opt_RS_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64],
                   float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    int i;
    // C channel
    SDQ::Block.ent.clear();
    SDQ::Block.P.clear();
    std::fill_n(SDQ::Block.state.ID, 64,0);
    std::fill_n(SDQ::Block.state.rs, 64,0);
    // initialize Pc0
    
    for(i=0; i<SDQ::seq_len_C; i++){
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], SDQ::RSlst, SDQ::IDlst);
        cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.ent);
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], SDQ::RSlst, SDQ::IDlst);
        cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.P);
    }
    SDQ::RSlst.clear(); SDQ::IDlst.clear();
    norm(SDQ::Block.P, SDQ::Block.ent);
    cal_ent(SDQ::Block.ent);
    SDQ::Block.set_channel('W');
    SDQ::Block.set_Q_table(SDQ::Q_table_C);
    SDQ::Loss = 0;
    SDQ::Block.state.ent=0;
    
    for(i=0; i<SDQ::seq_len_C; i++){
        std::fill_n(SDQ::Block.state.ID, 64,0);
        std::fill_n(SDQ::Block.state.rs, 64,0);
        SDQ::Block.cal_RS(seq_dct_coefs_Cb[i],
                          seq_dct_idxs_Cb[i],
                          SDQ::RSlst, SDQ::IDlst);
        RSlst_to_Block(seq_dct_idxs_Cb[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Cb[i]);
        SDQ::RSlst.clear();SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;
    }
    SDQ::Block.set_channel('X');
    
    for(i=0; i<SDQ::seq_len_C; i++){
        std::fill_n(SDQ::Block.state.ID, 64,0);
        std::fill_n(SDQ::Block.state.rs, 64,0);
        SDQ::Block.cal_RS(seq_dct_coefs_Cr[i],
                          seq_dct_idxs_Cr[i],
                          SDQ::RSlst, SDQ::IDlst);
        RSlst_to_Block(seq_dct_idxs_Cr[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Cr[i]);
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;  
    }
}

void SDQ::opt_Q_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    float divisor=0;
    float denominator=0;
    float val;
    int i,j;
    //TODO: start with 1

    
    for(j=1; j<64; j++){
        for(i=0; i<SDQ::seq_len_Y; i++){
            divisor += seq_dct_coefs_Y[i][j]*seq_dct_idxs_Y[i][j];
            denominator += pow(seq_dct_idxs_Y[i][j],2);
        }
        if(denominator > EPS){
            val = divisor/denominator;
            val = round(MinMaxClip(val, MINQVALUE, MAXQVALUE));
            SDQ::Q_table_Y[j] = val;
        }
        else    Q_table_Y[j] = MAXQVALUE;
        divisor=0; denominator=0; 
    }
}

void SDQ::opt_Q_C(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                  float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]){      
    float divisor=0;
    float denominator=0;
    float val;
    int i,j;
    
    for(j=1; j<64; j++){  
        for(i=0; i<SDQ::seq_len_C; i++){
            divisor += seq_dct_coefs_Cb[i][j]*seq_dct_idxs_Cb[i][j]*SDQ::Block.Sen_Map[1][j];
            divisor += seq_dct_coefs_Cr[i][j]*seq_dct_idxs_Cr[i][j]*SDQ::Block.Sen_Map[2][j];
            denominator += pow(seq_dct_idxs_Cb[i][j],2)*SDQ::Block.Sen_Map[1][j];
            denominator += pow(seq_dct_idxs_Cr[i][j],2)*SDQ::Block.Sen_Map[2][j];
        }
        if(denominator > EPS){
            val = divisor/denominator;
            val = round(MinMaxClip(val, MINQVALUE, MAXQVALUE));
            SDQ::Q_table_C[j] = val;
        }    
        else    Q_table_C[j] = MAXQVALUE;
        divisor=0;denominator=0; 
    }
}

void SDQ::opt_Q_Y_DC(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    float numerator=0;
    float denominator=0;
    float val;
    
    for(int i=0; i<seq_len_Y; i++){
        numerator += seq_dct_coefs_Y[i][0]*seq_dct_idxs_Y[i][0];
        denominator += pow(seq_dct_idxs_Y[i][0],2);
    }
    if(denominator > EPS){
        val = round(numerator/denominator);
        val = MinMaxClip(val, MINQVALUE, MAXQVALUE);
        Q_table_Y[0] = val;
    }
    else    Q_table_Y[0] = MAXQVALUE;
}

void SDQ::opt_Q_C_DC(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                     float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]){      
    float numerator=0;
    float denominator=0;
    float val;
    
    for(int i=0; i<seq_len_C; i++){
        numerator += seq_dct_coefs_Cb[i][0]*seq_dct_idxs_Cb[i][0]*Block.Sen_Map[1][0];
        numerator += seq_dct_coefs_Cr[i][0]*seq_dct_idxs_Cr[i][0]*Block.Sen_Map[2][0];
        denominator += pow(seq_dct_idxs_Cb[i][0],2)*Block.Sen_Map[1][0];
        denominator += pow(seq_dct_idxs_Cr[i][0],2)*Block.Sen_Map[2][0];
    }
    if(denominator > EPS){
        val = round(numerator/denominator);
        val = MinMaxClip(val, MINQVALUE, MAXQVALUE);
        Q_table_C[0] = val;
    }
    else    Q_table_C[0] = MAXQVALUE;
}

void SDQ::opt_DC_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    // calculate entropy rate for different size groups
    float ent[12];
    for(int s=0; s<12; s++){
        if(P_DC_Y[s]==0)    ent[s] = -log2(1/seq_len_Y)+s;
        else    ent[s] = -log2(P_DC_Y[s]/seq_len_Y)+s;
    }

    // DC trellis
    auto v = new float[seq_len_Y][33];
    auto cost_mini = new float[seq_len_Y][33];
    auto opt_pre = new int[seq_len_Y][33];     // store the j value of the optimal predecessor
    // set v values
    for(int i=0; i<seq_len_Y; i++){
        v[i][16] = round(seq_dct_coefs_Y[i][0]/Q_table_Y[0]);   // HDQ DC index
        for(int j=0; j<16; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
        for(int j=17; j<33; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
    }
    // initialize cost_mini[0][:]
    for(int j=0; j<33; j++){
        int s = size_group(v[0][j],11,0);
        cost_mini[0][j] = Beta_S*Block.Sen_Map[0][0]*pow(seq_dct_coefs_Y[0][0]-Q_table_Y[0]*v[0][j],2)+ent[s];
    }
    // figure out cost_mini and opt_pre for all states in all stages
    for(int i=1; i<seq_len_Y; i++){
        for(int j=0; j<33; j++){
            float comp_cost[33];
            for(int k=0; k<33; k++){    // k is j', which specifies the incoming path
                comp_cost[k] = cost_mini[i-1][k];
                int s = size_group(v[i][j]-v[i-1][k],11,0);   // calculate the size group of DIFF
                float cost_inc;
                cost_inc = Beta_S*Block.Sen_Map[0][0]*pow(seq_dct_coefs_Y[i][0]-Q_table_Y[0]*v[i][j],2)+ent[s];     // calculate cost_inc
                comp_cost[k] += cost_inc;   // add cost_inc to cost_mini
            }     
            cost_mini[i][j] = *min_element(comp_cost, comp_cost+33);
            opt_pre[i][j] = min_element(comp_cost, comp_cost+33)-comp_cost;
        }
    }

    // update DC indices
    int flag = min_element(cost_mini[seq_len_Y-1], cost_mini[seq_len_Y-1]+33)-cost_mini[seq_len_Y-1];
    seq_dct_idxs_Y[seq_len_Y-1][0] = v[seq_len_Y-1][flag];
    // backtrack
    for (int i=seq_len_Y-1; i > 0; i--) {
        seq_dct_idxs_Y[i-1][0] = v[i-1][opt_pre[i][flag]];
        flag = opt_pre[i][flag];
    }
    delete []v;
    delete []cost_mini;
    delete []opt_pre;
}

void SDQ::opt_DC_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64], 
                   float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    // calculate entropy rate for different size groups
    float ent[12];
    for(int s=0; s<12; s++){
        if(P_DC_C[s]==0)    ent[s] = -log2(1/seq_len_C)+s;
        else    ent[s] = -log2(P_DC_C[s]/seq_len_C)+s;
    }

    auto v = new float[seq_len_C][33];
    auto cost_mini = new float[seq_len_C][33];
    auto opt_pre = new int[seq_len_C][33];

    // Cb channel
    for(int i=0; i<seq_len_C; i++){
        v[i][16] = round(seq_dct_coefs_Cb[i][0]/Q_table_C[0]);
        for(int j=0; j<16; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
        for(int j=17; j<33; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
    }

    for(int j=0; j<33; j++){
        int s = size_group(v[0][j],11,0);
        cost_mini[0][j] = Beta_W*Block.Sen_Map[1][0]*pow(seq_dct_coefs_Cb[0][0]-Q_table_C[0]*v[0][j],2)+ent[s];
    }

    for(int i=1; i<seq_len_C; i++){
        for(int j=0; j<33; j++){
            float comp_cost[33];
            for(int k=0; k<33; k++){
                comp_cost[k] = cost_mini[i-1][k];
                int s = size_group(v[i][j]-v[i-1][k],11,0);
                float cost_inc;
                cost_inc = Beta_W*Block.Sen_Map[1][0]*pow(seq_dct_coefs_Cb[i][0]-Q_table_C[0]*v[i][j],2)+ent[s];
                comp_cost[k] += cost_inc;
            }     
            cost_mini[i][j] = *min_element(comp_cost, comp_cost+33);
            opt_pre[i][j] = min_element(comp_cost, comp_cost+33)-comp_cost;
        }
    }

    int flag = min_element(cost_mini[seq_len_C-1], cost_mini[seq_len_C-1]+33)-cost_mini[seq_len_C-1];
    seq_dct_idxs_Cb[seq_len_C-1][0] = v[seq_len_C-1][flag];
    for (int i=seq_len_C-1; i > 0; i--) {
        seq_dct_idxs_Cb[i-1][0] = v[i-1][opt_pre[i][flag]];
        flag = opt_pre[i][flag];
    }

    // Cr channel
    for(int i=0; i<seq_len_C; i++){
        v[i][16] = round(seq_dct_coefs_Cr[i][0]/Q_table_C[0]);
        for(int j=0; j<16; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
        for(int j=17; j<33; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
    }

    for(int j=0; j<33; j++){
        int s = size_group(v[0][j],11,0);
        cost_mini[0][j] = Beta_X*Block.Sen_Map[2][0]*pow(seq_dct_coefs_Cr[0][0]-Q_table_C[0]*v[0][j],2)+ent[s];
    }

    for(int i=1; i<seq_len_C; i++){
        for(int j=0; j<33; j++){
            float comp_cost[33];
            for(int k=0; k<33; k++){
                comp_cost[k] = cost_mini[i-1][k];
                int s = size_group(v[i][j]-v[i-1][k],11,0);
                float cost_inc;
                cost_inc = Beta_X*Block.Sen_Map[2][0]*pow(seq_dct_coefs_Cr[i][0]-Q_table_C[0]*v[i][j],2)+ent[s];
                comp_cost[k] += cost_inc;
            }     
            cost_mini[i][j] = *min_element(comp_cost, comp_cost+33);
            opt_pre[i][j] = min_element(comp_cost, comp_cost+33)-comp_cost;
        }
    }

    flag = min_element(cost_mini[seq_len_C-1], cost_mini[seq_len_C-1]+33)-cost_mini[seq_len_C-1];
    seq_dct_idxs_Cr[seq_len_C-1][0] = v[seq_len_C-1][flag];

    for (int i=seq_len_C-1; i > 0; i--) {
        seq_dct_idxs_Cr[i-1][0] = v[i-1][opt_pre[i][flag]];
        flag = opt_pre[i][flag];
    }
    delete []v;
    delete []cost_mini;
    delete []opt_pre;
}