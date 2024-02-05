#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    
    int linear_idx = x * (sizeZ * sizeY * sizeX) + y * (sizeZ * sizeY) + z * (sizeZ) + b;
    if (linear_idx >= tensor.size()){
        // std::cout << "fourDimRead ERROR" << std::endl;
        printf("fourDimRead ERROR, access: %i %i %i %i, tensor size:%zu, dim sizes: %i %i %i, linear_idx:%i\n", x,y,z,b,tensor.size(), sizeX, sizeY, sizeZ, linear_idx);
        exit(0);
    }
    return tensor[linear_idx];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    int linear_idx = x * (sizeZ * sizeY * sizeX) + y * (sizeZ * sizeY) + z * (sizeZ)+ b;
    if (linear_idx >= tensor.size()){
        std::cout << "fourDimWrite ERROR" << std::endl;
        exit(0);
    }
    tensor[linear_idx] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    printf("B:%i H:%i N:%i d:%i\n", B,H,N,d);
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b=0; b < B; b++) {
        for (int h=0; h < H; h++) {
            // Matrix multiply Q * K^t
            for (int i=0; i < N; i++) {
                for (int j=0; j < N; j++) {
                    int row_Q = i;
                    int row_K = j;
                    float res = 0;
                    for (int col=0; col < d; col++) {
                        float r1 = fourDimRead(Q,b,h,row_Q,col,H,N,d);
                        float r2 = fourDimRead(K,b,h,row_K,col,H,N,d);
                        res += r1 * r2;
                    }
                    twoDimWrite(QK_t, i, j, N, res);
                }
            }
            // Softmax
            for (int i=0; i < N; i++) {
                float exp_sum = 0;
                for (int j=0; j < N; j++) {
                    exp_sum += exp(twoDimRead(QK_t, i, j, N));
                }
                for (int j=0; j < N; j++) {
                    float res = exp(twoDimRead(QK_t, i, j, N)) / exp_sum;
                    twoDimWrite(QK_t, i, j, N, res);
                }
            }
            // Matrix multiply QK^t * V
            for (int i=0; i < N; i++) {
                for (int j=0; j < d; j++) {
                    float res = 0;
                    for (int k=0; k < N; k++) {
                        res += twoDimRead(QK_t, i, k, N) * twoDimRead(V, k, j, d);
                    }
                    fourDimWrite(O, b, h, i, j,H,N,d,res);
                }
            }
        }
    }    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //

    int BLOCKSIZE = 32; // cache line size
    printf("B:%i H:%i N:%i d:%i\n", B,H,N,d);

    for (int b=0; b < B; b++) {
        for (int h=0; h < H; h++) {
            int idiff;
            int jdiff;
            int kdiff;
            // Matrix multiply Q * K^t
            for (int iblock=0; iblock < N; iblock += BLOCKSIZE) {
                idiff = std::min(BLOCKSIZE, N-iblock);
                for (int jblock=0; jblock < N; jblock += BLOCKSIZE) {
                    jdiff = std::min(BLOCKSIZE, N-jblock);
                    for (int kblock=0; kblock < d; kblock += BLOCKSIZE) {
                        kdiff = std::min(BLOCKSIZE, d-kblock);
                        for (int i=0; i < idiff; i++) {
                            for (int j=0; j < jdiff; j++) {
                                int i_idx = iblock + i;
                                int j_idx = jblock + j;
                                float res = twoDimRead(QK_t, i_idx, j_idx, N);
                                for (int k=0; k < kdiff; k++) {
                                    int k_idx = kblock + k;
                                    float r1 = fourDimRead(Q, b, h, i_idx, k_idx, H, N, d);
                                    float r2 = fourDimRead(K, b, h, j_idx, k_idx, H, N, d);
                                    res += r1 * r2;
                                }
                                twoDimWrite(QK_t, i_idx, j_idx, N, res);
                            }
                        }   
                    }
                }
            }

            // Softmax
            for (int i=0; i < N; i++) {
                float exp_sum = 0;
                for (int j=0; j < N; j++) {
                    exp_sum += exp(twoDimRead(QK_t, i, j, N));
                }
                for (int j=0; j < N; j++) {
                    float res = exp(twoDimRead(QK_t, i, j, N)) / exp_sum;
                    twoDimWrite(QK_t, i, j, N, res);
                }
            }

            for (int iblock=0; iblock < N; iblock += std::min(BLOCKSIZE, N-iblock)) {
                idiff = std::min(BLOCKSIZE, N-iblock);
                for (int jblock=0; jblock < d; jblock += std::min(BLOCKSIZE, d-jblock)) {
                    jdiff = std::min(BLOCKSIZE, N-jblock);
                    for (int kblock=0; kblock < N; kblock += std::min(BLOCKSIZE, N-kblock)) {
                        kdiff = std::min(BLOCKSIZE, N-kblock);
                        for (int i=0; i < idiff; i++) {
                            for (int j=0; j < jdiff; j++) {
                                int i_idx = iblock + i;
                                int j_idx = jblock + j;
                                float res = fourDimRead(O, b, h, i_idx, j_idx, H, N, d);
                                for (int k=0; k < kdiff; k++) {
                                    int k_idx = kblock + k; 
                                    res += twoDimRead(QK_t, i_idx, k_idx, N) * twoDimRead(V, k_idx, j_idx, d);
                                }
                                fourDimWrite(O, b, h, i_idx, j_idx, H, N, d, res);
                            }
                        }
                    }
                }
            }
        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {

        //loop over heads
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N ; i++) {

		        // YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
                int zero_idx = 0;
                
                // Matrix mul
                for (int j=0; j < N; j++) {
                    int row_Q = i;
                    int row_K = j;
                    float res = 0;
                    for (int col=0; col < d; col++) {
                        float r1 = fourDimRead(Q,b,h,row_Q,col,H,N,d);
                        float r2 = fourDimRead(K,b,h,row_K,col,H,N,d);
                        res += r1 * r2;
                    }
                    twoDimWrite(ORow, zero_idx, j, N, res);
                }

                // Softmax
                float exp_sum = 0;
                for (int j=0; j < N; j++) {
                    exp_sum += exp(twoDimRead(ORow, zero_idx, j, N));
                }
                for (int j=0; j < N; j++) {
                    float res = exp(twoDimRead(ORow, zero_idx, j, N)) / exp_sum;
                    twoDimWrite(ORow, zero_idx, j, N, res);
                }
                
                // Matrix mul
                for (int j=0; j < d; j++) {
                    float res = 0;
                    for (int k=0; k < N; k++) {
                        res += twoDimRead(ORow, zero_idx, k, N) * twoDimRead(V, k, j, d);
                    }
                    fourDimWrite(O, b, h, i, j,H,N,d,res);
                }
            }
	    }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
