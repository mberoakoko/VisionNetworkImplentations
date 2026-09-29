from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from loguru import logger

from data.data_loader import load_mnist_images, load_mnist_labels

_DEFAULT_KEY: PRNGKeyArray = jax.random.key(1000)

import matplotlib.pyplot as plt


class PPCAStateItern(NamedTuple):
    w: Array 
    mu: Array 
    sigma: Array 
    iter: int 
    d_w: float 
    d_sig: float


def display_results(X_sub: np.ndarray, X_recon: Array, y_sub: np.ndarray , Z: Array) -> None :
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Originals (Top) vs. Reconstructions (Bottom)", fontsize=16)

    for i in range(5):
        # Original
        axes[0, i].imshow(X_sub[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
    
        # Reconstructed
        # Clip values to [0, 1] for valid image rendering
        recon_img = np.clip(X_recon[i], 0, 1) 
        axes[1, i].imshow(recon_img.reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Plot the 2D Latent Space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y_sub, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("MNIST Latent Space (Z)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()

@jax.jit 
def compute_elbo(X_centered: Array, W: Array, sigma_2: float, E_z: Array, E_zzT: Array, m_inv: Array) -> float:
    _, D = X_centered.shape 
    latent_dim = W.shape[1]

    # Expected reconstruction of the log_likelihood
    tr_W_W_EzzT = jnp.trace(W.T @ W @ E_zzT )
    expected_sq_err = (jnp.mean(jnp.sum(X_centered**2, axis=1)) 
                       - 2.0 * jnp.mean(jnp.sum(X_centered * (E_z @ W.T), axis=1)) 
                       + tr_W_W_EzzT)
    expected_log_lik = -0.5 * D * (jnp.log(2.0 * jnp.pi) + jnp.log(sigma_2)) - (0.5 / sigma_2) * expected_sq_err

    # K_L divergence
    Sigma_q = sigma_2 * m_inv
    trace_term = jnp.trace(Sigma_q)
    mean_sq_term = jnp.mean(jnp.sum(E_z**2, axis=1))
    log_det_Sigma = jnp.linalg.slogdet(Sigma_q)[1]
    
    kl_div = 0.5 * (trace_term + mean_sq_term - latent_dim - log_det_Sigma)
    
    # ELBO = Reconstruction - Penalty
    return expected_log_lik - kl_div

@jax.jit 
def ppca_em_step_fit(X: Array, latent_dim: int =2, max_iters: int  = 1000, tol: float =1e-5, key: PRNGKeyArray = _DEFAULT_KEY) -> PPCAStateItern:
    N, D = X.shape 

    mu = jnp.mean(X, axis=0)
    X_centered = X - mu 

    w_key, _ = jax.random.split(key)
    w_init = jax.random.normal(w_key, shape=(D, latent_dim)) * 0.1
    # w_init = X_centered[:latent_dim].T
    sigma_2_init = jnp.var(X_centered) / 0.5

    init_state: PPCAStateItern = PPCAStateItern( w_init, mu,  sigma_2_init, 0, 1.0, 1.0) 

    def cond_func(state: PPCAStateItern) -> bool:
        _, _, _,  iterations, d_w, d_sig = state 
        return (iterations <= max_iters) & ( ( d_w >= tol )  | ( d_sig >= tol )  )

    def body_func(state: PPCAStateItern) -> PPCAStateItern:
        w_old,mu,  sigma_old, curr_iter, _, _ = state 

        M = w_old.T @ w_old + sigma_old * jnp.eye(latent_dim)
        m_inv = jnp.linalg.inv(M)

        # Expectation step
        E_z = X_centered @ w_old @ m_inv 
        E_zzT = sigma_old * m_inv + (E_z.T @ E_z) / N 


        # Maximization step 
        Sum_xz = X_centered.T @ E_z
        W_new = Sum_xz @ jnp.linalg.inv(E_zzT * N)
        
        reconstruction_term = jnp.sum(X_centered**2) - jnp.sum(Sum_xz * W_new)
        sigma2_new = reconstruction_term / (N * D)
        
        diff_W = jnp.linalg.norm(W_new - w_old)
        diff_sigma2: float = jnp.abs(sigma2_new - sigma_old)
        return PPCAStateItern( W_new, mu,  sigma2_new, curr_iter + 1, diff_W, diff_sigma2 ) 

    return jax.lax.while_loop(cond_func, body_func, init_state)


@jax.jit 
def encode(x: Array, w: Array, sigma2: float, mu: float):
    x_centered = x - mu 
    latent_dim = w.shape[1]
    m = w.T @ w + sigma2 * jnp.eye(latent_dim)
    m_inv = jnp.linalg.inv(m)
    return x_centered @ w @ m_inv

@jax.jit
def decode(Z: Array , W: Array , mu: Array):
    return Z @ W.T + mu 



def main() -> None:
    x_images = load_mnist_images()
    y = load_mnist_labels()
    print(f"{x_images.shape=}")

    x_all = x_images.reshape(-1, 28*28).astype(np.float32) / 255.0 
    print(x_all.shape)
    

    state_result: PPCAStateItern = ppca_em_step_fit(jnp.array(x_all), max_iters=2000)
    
    N_samples = 15000
    X_sub = x_all[:N_samples, :]
    y_sub = y[:N_samples]

    logger.info("Encoding and decoding")
    z = encode(X_sub, state_result.w, state_result.sigma, state_result.mu)
    x_recon = decode(z, state_result.w, state_result.mu)

    display_results(X_sub, x_recon, y_sub, z)



if __name__ == "__main__":
    main()
