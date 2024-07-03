#include <map>
#include <ctime>
#include <chrono>
#include "../Utils/utils.h"
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"
using namespace std;
const float MIN_Q_VAL = 1;
class DCT_BPP{
    public:
        // attributes
        float Q_table_Y[64];
        float Q_table_C[64];
        int seq_len_Y, seq_len_C; // # 8x8 DCT blocks after subsampling
        int n_row;
        int n_col;
        // BLOCK Block;
        int seq_block_dcts[64];
        int DCT_block_shape[3];
        int img_shape_Y[2], img_shape_C[2]; // size of channels after subsampling
        //TODO: P_DC for DC coefficient
        map<int, float> P_DC_Y;
        map<int, float> P_DC_C;
        float J_Y = 10e10;
        float J_C = 10e10;
        int QF_C;
        int QF_Y;
        int J, a, b;
        float Loss;
        float EntACY = 0;
        float EntACC = 0;
        float EntDCY = 0;
        float EntDCC = 0;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__();
        float __call__(vector<vector<vector<float>>>& image);
};

void DCT_BPP::__init__(){
    DCT_BPP::RSlst.reserve(64);
    DCT_BPP::IDlst.reserve(64);
    DCT_BPP::RSlst.clear();
    DCT_BPP::IDlst.clear();
    DCT_BPP::J_Y = 10e10;
    DCT_BPP::J_C = 10e10;
    // No down sampling for padding
    DCT_BPP::J = 4;
    DCT_BPP::a = 4;
    DCT_BPP::b = 4;
}

float DCT_BPP::__call__(vector<vector<vector<float>>>& image){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, DCT_BPP::img_shape_Y);
    DCT_BPP::n_col = DCT_BPP::img_shape_Y[1];
    DCT_BPP::n_row = DCT_BPP::img_shape_Y[0];
    DCT_BPP::seq_len_Y = pad_shape(DCT_BPP::img_shape_Y[0], 8)*pad_shape(DCT_BPP::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(DCT_BPP::img_shape_Y[0], J);
    int pad_cols = pad_shape(DCT_BPP::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    DCT_BPP::img_shape_C[0] = DCT_BPP::img_shape_Y[0];
    DCT_BPP::img_shape_C[1] = DCT_BPP::img_shape_Y[1];
    DCT_BPP::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;

    // the input is the direct indices, no need for extra things
    auto blockified_img_Y = new float[DCT_BPP::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[DCT_BPP::seq_len_C][8][8];
    auto blockified_img_Cr = new float[DCT_BPP::seq_len_C][8][8];

    auto seq_dct_idxs_Y = new float[DCT_BPP::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[DCT_BPP::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[DCT_BPP::seq_len_C][64];

    auto DC_idxs_Y = new float[DCT_BPP::seq_len_Y];
    auto DC_idxs_Cb = new float[DCT_BPP::seq_len_C];
    auto DC_idxs_Cr = new float[DCT_BPP::seq_len_C];

    blockify(image[0], DCT_BPP::img_shape_Y, blockified_img_Y);
    blockify(image[1], DCT_BPP::img_shape_C, blockified_img_Cb);    
    blockify(image[2], DCT_BPP::img_shape_C, blockified_img_Cr);

    block_2_seq(blockified_img_Y, seq_dct_idxs_Y, DCT_BPP::seq_len_Y);
    block_2_seq(blockified_img_Cb, seq_dct_idxs_Cb, DCT_BPP::seq_len_C);
    block_2_seq(blockified_img_Cr, seq_dct_idxs_Cr, DCT_BPP::seq_len_C);

    map<int, float> DC_P;
    map<int, float> AC_Y_P;
    map<int, float> AC_C_P;
    DC_P.clear();
    float EntDCY=0;
    float EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, DCT_BPP::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, DCT_BPP::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, DCT_BPP::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, DCT_BPP::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, DCT_BPP::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, DCT_BPP::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    for(i=0; i<DCT_BPP::seq_len_Y; i++){
        DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], DCT_BPP::RSlst, DCT_BPP::IDlst);
        cal_P_from_RSlst(DCT_BPP::RSlst, AC_Y_P);
    }
    DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    AC_Y_P.clear();

    for(i=0; i<DCT_BPP::seq_len_C; i++){
        DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], DCT_BPP::RSlst, DCT_BPP::IDlst);
        cal_P_from_RSlst(DCT_BPP::RSlst, AC_C_P);
        DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], DCT_BPP::RSlst, DCT_BPP::IDlst);
        cal_P_from_RSlst(DCT_BPP::RSlst, AC_C_P);
    }
    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE;// Run_length coding
    BPP = file_size/DCT_BPP::img_shape_Y[0]/DCT_BPP::img_shape_Y[1]; // REMOVE THIS KAIXIANG
    // delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    return BPP;
}
