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
        // Quantization tables for luminance and chrominance
        float Q_table_Y[64];
        float Q_table_C[64];

        // Sequence lengths for Y and C components after DCT and subsampling
        int seq_len_Y, seq_len_C; // 8x8 DCT blocks after subsampling

        // Image dimensions in rows and columns
        int n_row;
        int n_col;

        // Sequence of DCT block indices
        int seq_block_dcts[64];

        // DCT block shape as 3-dimensional array (presumed dimensions)
        int DCT_block_shape[3];

        // Dimensions of the Y and C components after subsampling
        int img_shape_Y[2], img_shape_C[2]; // size of channels after subsampling

        // Probabilities of DC components for Y and C
        map<int, float> P_DC_Y;
        map<int, float> P_DC_C;

        // Initialization of cost functions for Y and C components to a high value
        float J_Y = 10e10;
        float J_C = 10e10;

         // Quantization factors for chrominance and luminance
        int QF_C;
        int QF_Y;

        // Parameters for further processing
        int J, a, b;

        // Loss variable, perhaps for tracking quantization or compression loss
        float Loss;

        // Entropy values for AC and DC components of Y and C
        float EntACY = 0;
        float EntACC = 0;
        float EntDCY = 0;
        float EntDCC = 0;

        // Run-length and ID lists, presumably for entropy coding
        vector<int> RSlst;
        vector<int> IDlst;

        // Constructor declaration
        void __init__();

        // Callable method that processes an image and returns bits per pixel (BPP)
        float __call__(vector<vector<vector<float>>>& image);
};

void DCT_BPP::__init__(){
    // Reserve space for run-length and ID lists, and reset their values
    DCT_BPP::RSlst.reserve(64);
    DCT_BPP::IDlst.reserve(64);
    DCT_BPP::RSlst.clear();
    DCT_BPP::IDlst.clear();

    // Reset cost functions to a high value
    DCT_BPP::J_Y = 10e10;
    DCT_BPP::J_C = 10e10;

    // No down sampling for padding
    DCT_BPP::J = 4;
    DCT_BPP::a = 4;
    DCT_BPP::b = 4;
}

float DCT_BPP::__call__(vector<vector<vector<float>>>& image){
    // Temporary variables for loops and indexing
    int i,j,k,l,r,s;

    // Bit array to store bit usage statistics
    int BITS[33];
    int size;

    // Initialize BITS array to zero
    std::fill_n(BITS, 33, 0);

    // Determine shape of the image for Y component
    shape(image, DCT_BPP::img_shape_Y);
    
    // Store dimensions of Y component
    DCT_BPP::n_col = DCT_BPP::img_shape_Y[1];
    DCT_BPP::n_row = DCT_BPP::img_shape_Y[0];

    // Calculate sequence length for Y component based on padding and block size
    DCT_BPP::seq_len_Y = pad_shape(DCT_BPP::img_shape_Y[0], 8) * pad_shape(DCT_BPP::img_shape_Y[1], 8)/64;

    // Horizontal and vertical sampling steps based on parameters a and b
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}

    // Calculate padded dimensions
    int pad_rows = pad_shape(DCT_BPP::img_shape_Y[0], J);
    int pad_cols = pad_shape(DCT_BPP::img_shape_Y[1], J);

    // Calculate sampled dimensions
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;

    // Set chrominance dimensions equal to luminance dimensions
    DCT_BPP::img_shape_C[0] = DCT_BPP::img_shape_Y[0];
    DCT_BPP::img_shape_C[1] = DCT_BPP::img_shape_Y[1];

    // Calculate sequence length for C components based on subsampling
    DCT_BPP::seq_len_C = pad_shape(Smplcols, 8) * pad_shape(Smplrows, 8)/64;

    // the input is the direct indices, no need for extra things
    auto blockified_img_Y = new float[DCT_BPP::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[DCT_BPP::seq_len_C][8][8];
    auto blockified_img_Cr = new float[DCT_BPP::seq_len_C][8][8];

    auto seq_dct_idxs_Y = new float[DCT_BPP::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[DCT_BPP::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[DCT_BPP::seq_len_C][64];

     // Allocate memory for DC indices
    auto DC_idxs_Y = new float[DCT_BPP::seq_len_Y];
    auto DC_idxs_Cb = new float[DCT_BPP::seq_len_C];
    auto DC_idxs_Cr = new float[DCT_BPP::seq_len_C];

    // Blockify input image channels into smaller DCT blocks
    blockify(image[0], DCT_BPP::img_shape_Y, blockified_img_Y);
    blockify(image[1], DCT_BPP::img_shape_C, blockified_img_Cb);    
    blockify(image[2], DCT_BPP::img_shape_C, blockified_img_Cr);

    // Convert blockified images into sequences of DCT indices
    block_2_seq(blockified_img_Y, seq_dct_idxs_Y, DCT_BPP::seq_len_Y);
    block_2_seq(blockified_img_Cb, seq_dct_idxs_Cb, DCT_BPP::seq_len_C);
    block_2_seq(blockified_img_Cr, seq_dct_idxs_Cr, DCT_BPP::seq_len_C);

    // Clear and calculate DC component probabilities and entropies
    map<int, float> DC_P;
    map<int, float> AC_Y_P;
    map<int, float> AC_C_P;
    DC_P.clear();

    // Initialize entropy variables for DC components of Y and C
    float EntDCY=0;
    float EntDCC=0;

    // Apply Differential Pulse Code Modulation on the Y component's DCT indices to get DC indices
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, DCT_BPP::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, DCT_BPP::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    // Calculate the Huffman code size based on the DC probabilities for Y component
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    // Repeat DPCM and probability calculations for Cb component
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, DCT_BPP::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, DCT_BPP::seq_len_C);
    // Repeat DPCM and probability calculations for Cr component
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, DCT_BPP::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, DCT_BPP::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    // Process AC components for Y using run-length and index lists
    for(i=0; i<DCT_BPP::seq_len_Y; i++){
        DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
        // Convert block DCT indices to run-length sequences and indices for Y component
        Block_to_RSlst(seq_dct_idxs_Y[i], DCT_BPP::RSlst, DCT_BPP::IDlst);
        // Calculate probabilities from the run-length sequences for Y component
        cal_P_from_RSlst(DCT_BPP::RSlst, AC_Y_P);
    }
    // Clear lists and remove total key from AC probabilities, then calculate entropy
    DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    // Clear AC probability map after use
    AC_Y_P.clear();

    // Process AC components for C components (Cb and Cr)
    for(i=0; i<DCT_BPP::seq_len_C; i++){
        DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
        // Process Cr component
        Block_to_RSlst(seq_dct_idxs_Cr[i], DCT_BPP::RSlst, DCT_BPP::IDlst);
        cal_P_from_RSlst(DCT_BPP::RSlst, AC_C_P);
        // Clear lists before processing Cb component
        DCT_BPP::RSlst.clear(); DCT_BPP::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], DCT_BPP::RSlst, DCT_BPP::IDlst);
        cal_P_from_RSlst(DCT_BPP::RSlst, AC_C_P);
    }
    // Clear lists and remove total key, calculate entropy for C components
    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);

    // Calculate bits per pixel (BPP) based on the total entropy for all components and additional data (flag size)
    float BPP=0;
    float file_size = EntACC + EntACY + EntDCC + EntDCY + FLAG_SIZE; // Run_length coding
    BPP = file_size/DCT_BPP::img_shape_Y[0]/DCT_BPP::img_shape_Y[1]; // REMOVE THIS KAIXIANG

    // delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;

    // Return calculated BPP value
    return BPP;
}
