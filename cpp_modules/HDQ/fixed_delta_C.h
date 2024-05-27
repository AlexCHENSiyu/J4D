#include <map>
#include <ctime>
#include <chrono>
#include "../Utils/utils.h"
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"
using namespace std;
const float MIN_Q_VAL = 1;
class fixed_delta_C{
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
        float delta_C = 0;
        int dct_index = 0;
        float Loss;
        float EntACY = 0;
        float EntACC = 0;
        float EntDCY = 0;
        float EntDCC = 0;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(int QF_Y, int QF_C, 
                      int J, int a, int b, float delta_C, int dct_index);
        float __call__(vector<vector<vector<float>>>& image);
};

void fixed_delta_C::__init__(int QF_Y, int QF_C, 
                   int J, int a, int b, float delta_C, int dct_index){
    fixed_delta_C::RSlst.reserve(64);
    fixed_delta_C::IDlst.reserve(64);
    fixed_delta_C::RSlst.clear();
    fixed_delta_C::IDlst.clear();
    quantizationTable(QF_Y, true, fixed_delta_C::Q_table_Y);
    quantizationTable(QF_C, false, fixed_delta_C::Q_table_C);
    fixed_delta_C::J_Y = 10e10;
    fixed_delta_C::J_C = 10e10;
    fixed_delta_C::J = J;
    fixed_delta_C::a = a;
    fixed_delta_C::b = b;
    fixed_delta_C::delta_C = delta_C;
    fixed_delta_C::dct_index = dct_index;
}

float fixed_delta_C::__call__(vector<vector<vector<float>>>& image){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, fixed_delta_C::img_shape_Y);
    fixed_delta_C::n_col = fixed_delta_C::img_shape_Y[1];
    fixed_delta_C::n_row = fixed_delta_C::img_shape_Y[0];
    fixed_delta_C::seq_len_Y = pad_shape(fixed_delta_C::img_shape_Y[0], 8)*pad_shape(fixed_delta_C::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(fixed_delta_C::img_shape_Y[0], J);
    int pad_cols = pad_shape(fixed_delta_C::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], fixed_delta_C::img_shape_Y, fixed_delta_C::img_shape_C, fixed_delta_C::J, fixed_delta_C::a, fixed_delta_C::b);
    Subsampling(image[2], fixed_delta_C::img_shape_Y, fixed_delta_C::img_shape_C, fixed_delta_C::J, fixed_delta_C::a, fixed_delta_C::b);

    fixed_delta_C::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new float[fixed_delta_C::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[fixed_delta_C::seq_len_C][8][8];
    auto blockified_img_Cr = new float[fixed_delta_C::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new float[fixed_delta_C::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new float[fixed_delta_C::seq_len_C][64];
    auto seq_dct_coefs_Cr = new float[fixed_delta_C::seq_len_C][64];

    auto seq_dct_idxs_Y = new float[fixed_delta_C::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[fixed_delta_C::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[fixed_delta_C::seq_len_C][64];

    auto DC_idxs_Y = new float[fixed_delta_C::seq_len_Y];
    auto DC_idxs_Cb = new float[fixed_delta_C::seq_len_C];
    auto DC_idxs_Cr = new float[fixed_delta_C::seq_len_C];

    blockify(image[0], fixed_delta_C::img_shape_Y, blockified_img_Y);
    blockify(image[1], fixed_delta_C::img_shape_C, blockified_img_Cb);    
    blockify(image[2], fixed_delta_C::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, fixed_delta_C::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, fixed_delta_C::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, fixed_delta_C::seq_len_C);

    // fixed_delta_C::delta_C

    // Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
    //          fixed_delta_C::Q_table_Y, fixed_delta_C::seq_len_Y);
    // Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
    //          fixed_delta_C::Q_table_C,fixed_delta_C::seq_len_C);
    // Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
    //          fixed_delta_C::Q_table_C,fixed_delta_C::seq_len_C);


    add_Distortion(seq_dct_coefs_Y, seq_dct_idxs_Y, fixed_delta_C::seq_len_Y, fixed_delta_C::delta_C, fixed_delta_C::dct_index);
    add_Distortion(seq_dct_coefs_Cb, seq_dct_idxs_Cb, fixed_delta_C::seq_len_C, fixed_delta_C::delta_C, fixed_delta_C::dct_index);
    add_Distortion(seq_dct_coefs_Cr, seq_dct_idxs_Cr, fixed_delta_C::seq_len_C, fixed_delta_C::delta_C, fixed_delta_C::dct_index);
    
    map<int, float> DC_P;
    map<int, float> AC_Y_P;
    map<int, float> AC_C_P;
    DC_P.clear();
    float EntDCY=0;
    float EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, fixed_delta_C::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, fixed_delta_C::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, fixed_delta_C::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, fixed_delta_C::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, fixed_delta_C::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, fixed_delta_C::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    for(i=0; i<fixed_delta_C::seq_len_Y; i++){
        fixed_delta_C::RSlst.clear(); fixed_delta_C::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], fixed_delta_C::RSlst, fixed_delta_C::IDlst);
        cal_P_from_RSlst(fixed_delta_C::RSlst, AC_Y_P);
    }
    fixed_delta_C::RSlst.clear(); fixed_delta_C::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    AC_Y_P.clear();

    for(i=0; i<fixed_delta_C::seq_len_C; i++){
        fixed_delta_C::RSlst.clear(); fixed_delta_C::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], fixed_delta_C::RSlst, fixed_delta_C::IDlst);
        cal_P_from_RSlst(fixed_delta_C::RSlst, AC_C_P);
        fixed_delta_C::RSlst.clear(); fixed_delta_C::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], fixed_delta_C::RSlst, fixed_delta_C::IDlst);
        cal_P_from_RSlst(fixed_delta_C::RSlst, AC_C_P);
    }
    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE;// Run_length coding
    BPP = file_size/fixed_delta_C::img_shape_Y[0]/fixed_delta_C::img_shape_Y[1]; // REMOVE THIS KAIXIANG
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    // Dequantize(seq_dct_idxs_Y, fixed_delta_C::Q_table_Y, fixed_delta_C::seq_len_Y); //seq_dct_idxs_Y: [][64]
    // Dequantize(seq_dct_idxs_Cb, fixed_delta_C::Q_table_C, fixed_delta_C::seq_len_C);
    // Dequantize(seq_dct_idxs_Cr, fixed_delta_C::Q_table_C, fixed_delta_C::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, fixed_delta_C::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, fixed_delta_C::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, fixed_delta_C::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], fixed_delta_C::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], fixed_delta_C::img_shape_C);
    deblockify(blockified_img_Cr, image[2], fixed_delta_C::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], fixed_delta_C::img_shape_Y, fixed_delta_C::J, fixed_delta_C::a, fixed_delta_C::b);
    Upsampling(image[2], fixed_delta_C::img_shape_Y, fixed_delta_C::J, fixed_delta_C::a, fixed_delta_C::b);
    return BPP;
}
