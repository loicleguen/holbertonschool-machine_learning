#!/usr/bin/env python3
"""
Optimization of Deep Learning Hyperparameters via Bayesian Methods (GPyOpt)
"""
import os
import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import matplotlib.pyplot as plt

# Masquage des logs verbeux de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class KerasBayesianOptimizer:
    """
    Handles dataset generation, model compilation, and Bayesian optimization
    for hyperparameter tuning.
    """

    def __init__(self, num_samples=1200, input_dim=16):
        """
        Initializes seeds, generates synthetic data, and sets search bounds.
        """
        np.random.seed(100)
        tf.random.set_seed(100)

        self.input_dim = input_dim
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Dataset synthétique pour un entraînement rapide
        X = np.random.normal(size=(num_samples, input_dim)).astype(np.float32)
        y = (np.sum(X[:, :3], axis=1, keepdims=True) > 0).astype(np.float32)

        # Separation Train (80%) / Validation (20%)
        split_idx = int(num_samples * 0.8)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]

        # Espace de recherche des 5 hyperparamètres
        self.bounds = [
            {
                'name': 'learning_rate',
                'type': 'continuous',
                'domain': (1e-4, 1e-2)
            },
            {
                'name': 'units',
                'type': 'discrete',
                'domain': (32, 64, 128, 256)
            },
            {
                'name': 'dropout',
                'type': 'continuous',
                'domain': (0.1, 0.5)
            },
            {
                'name': 'l2_weight',
                'type': 'continuous',
                'domain': (1e-4, 1e-2)
            },
            {
                'name': 'batch_size',
                'type': 'discrete',
                'domain': (32, 64, 128)
            }
        ]

    def _build_architecture(self, lr, units, dropout, l2_weight):
        """
        Constructs and compiles the Keras sequential model.
        """
        model = models.Sequential([
            layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_weight),
                input_shape=(self.input_dim,)
            ),
            layers.Dropout(dropout),
            layers.Dense(
                units // 2,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_weight)
            ),
            layers.Dropout(dropout),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def evaluate_candidate(self, x):
        """
        Objective function evaluated by GPyOpt to minimize validation loss.
        """
        params = x[0]
        lr = float(params[0])
        units = int(params[1])
        dropout = float(params[2])
        l2_weight = float(params[3])
        batch_size = int(params[4])

        model = self._build_architecture(lr, units, dropout, l2_weight)

        ckpt_filename = (
            f"model_lr_{lr:.4f}_units_{units}_drop_{dropout:.2f}_"
            f"l2_{l2_weight:.4f}_bs_{batch_size}.keras"
        )
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_filename)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path, monitor='val_loss',
                save_best_only=True, verbose=0
            )
        ]

        history = model.fit(
            self.X_train, self.y_train,
            epochs=20,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=0
        )

        min_val_loss = float(np.min(history.history['val_loss']))
        return min_val_loss

    def optimize(self, iterations=30):
        """
        Runs the Bayesian Optimization process and saves the final outputs.
        """
        opt = GPyOpt.methods.BayesianOptimization(
            f=self.evaluate_candidate,
            domain=self.bounds,
            acquisition_type='EI',
            exact_feval=True
        )

        opt.run_optimization(max_iter=iterations)

        best_x = opt.x_opt
        best_loss = opt.fx_opt

        # Écriture du fichier de rapport bayes_opt.txt
        with open('bayes_opt.txt', 'w') as f:
            f.write("=========================================\n")
            f.write("   Bayesian Optimization Final Report\n")
            f.write("=========================================\n\n")
            f.write(f"Best Target Loss (val_loss): {best_loss:.6f}\n\n")
            f.write("Optimized Hyperparameters:\n")
            f.write(f"  - Learning Rate : {best_x[0]:.6f}\n")
            f.write(f"  - Hidden Units  : {int(best_x[1])}\n")
            f.write(f"  - Dropout Rate  : {best_x[2]:.4f}\n")
            f.write(f"  - L2 Weight     : {best_x[3]:.6f}\n")
            f.write(f"  - Batch Size    : {int(best_x[4])}\n")

        # Génération du graphique de convergence
        opt.plot_convergence(filename='bayes_opt_convergence.png')
        plt.close('all')

        print("Optimisation terminée !")
        print(f"Meilleure val_loss : {best_loss:.4f}")


if __name__ == '__main__':
    tuner = KerasBayesianOptimizer()
    tuner.optimize(iterations=30)
