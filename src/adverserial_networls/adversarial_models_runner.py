import equinox as eqx
import jax.random
import optax

from adverserial_networls.model import ConvolutionalNeuralNetwork, TestDataGenerator
from adverserial_networls.train import (
    adversarial_train_step_builder,
    evaluate_batch,
    pgd_attack,
    run_training_epoch,
    standard_train_step,
)
from adverserial_networls.visualizer import plot_interactive_batch
from src.data.grain_data_loader import (
    GrainDatasetAdapter,
    adapt_grain_batch,
    get_test_loader,
    get_train_loader,
)


def attack_loss() -> None:
    key = jax.random.PRNGKey(0)

    model_key, tensor_key = jax.random.split(key)
    small, medium, large = TestDataGenerator.create(key=tensor_key, batch_size=10)
    model = ConvolutionalNeuralNetwork(num_classes=10, rng_key=model_key)
    result = pgd_attack(
        model=model,
        image=small.images,
        label=small.labels
    )
    plot_interactive_batch(
        model=model,
        images=small.images,
        pert=result,
        labels=small.labels
    )
    
def dummy_training_run():
    key = jax.random.PRNGKey(42)
    key_model, key_data, key_test = jax.random.split(key, 3)

    # 1. Hyperparameters & Model Setup
    batch_size = 16
    num_classes = 10
    learning_rate = 1e-3
    num_epochs = 50

    model = ConvolutionalNeuralNetwork(num_classes=num_classes, rng_key=key_model)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 2. Data Generator Helper
    def dummy_data_loader():
        nonlocal key_data
        key_data, k1, k2 = jax.random.split(key_data, 3)
        images = jax.random.uniform(k1, shape=(batch_size, 1, 32, 32))
        labels = jax.random.randint(
            k2, shape=(batch_size,), minval=0, maxval=num_classes
        )
        return type("Batch", (), {"images": images, "labels": labels})()

    print("--- Starting Standard Training ---")
    for epoch in range(1, num_epochs + 1):
        model, opt_state, avg_loss = run_training_epoch(
            model,
            opt_state,
            optimizer,
            dummy_data_loader,
            standard_train_step,
            num_steps=20,
        )
        print(f"Epoch {epoch:2d} | Avg Clean Loss: {avg_loss:.4f}", end="\r")

    print("\n--- Starting Adversarial Fine-Tuning ---")
    adv_step_fn = adversarial_train_step_builder(pgd_attack)
    for epoch in range(1, num_epochs + 1):
        model, opt_state, avg_loss = run_training_epoch(
            model, opt_state, optimizer, dummy_data_loader, adv_step_fn, num_steps=20
        )
        print(f"Epoch {epoch:2d} | Avg Adv Loss:   {avg_loss:.4f}")

    # 3. Test & Interactively Visualize Results
    test_batch = dummy_data_loader()
    pert = pgd_attack(model, test_batch.images, test_batch.labels, epsilon=0.1)

    clean_loss, clean_acc = evaluate_batch(model, test_batch.images, test_batch.labels)
    adv_loss, adv_acc = evaluate_batch(
        model, test_batch.images + pert, test_batch.labels
    )

    print(f"\nFinal Evaluation on Test Batch:")
    print(f"  Clean Accuracy:       {clean_acc:.1%}")
    print(f"  Adversarial Accuracy: {adv_acc:.1%}")

    # Launch GUI Visualizer
    plot_interactive_batch(model, test_batch.images, pert, test_batch.labels)
    
    
def real_training_run():
    key = jax.random.PRNGKey(42)

    # 1. Hyperparameters & Model Setup
    num_classes = 10
    learning_rate = 1e-3
    num_epochs = 20

    model = ConvolutionalNeuralNetwork(num_classes=num_classes, rng_key=key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print("--- Starting Standard Training with Grain ---")
    train_loader = get_train_loader()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_batches = 0

        # Wrap Grain iterator with the adapter
        for batch in GrainDatasetAdapter(train_loader):
            model, opt_state, loss_val = standard_train_step(
                model, opt_state, optimizer, batch.images, batch.labels
            )
            total_loss += float(loss_val)
            num_batches += 1

        print(f"Epoch {epoch:2d} | Avg Loss: {total_loss / num_batches:.4f}")

    print("\n--- Starting Adversarial Fine-Tuning with Grain ---")
    adv_step_fn = adversarial_train_step_builder(pgd_attack)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for batch in GrainDatasetAdapter(train_loader):
            model, opt_state, loss_val = adv_step_fn(
                model, opt_state, optimizer, batch.images, batch.labels, epsilon=0.1
            )
            total_loss += float(loss_val)
            num_batches += 1

        print(f"Epoch {epoch:2d} | Avg Adv Loss: {total_loss / num_batches:.4f}")

    # 3. Final Evaluation on Test Set
    test_loader = get_test_loader()
    test_batch = adapt_grain_batch(next(iter(test_loader)))

    pert = pgd_attack(model, test_batch.images, test_batch.labels, epsilon=0.1)

    clean_loss, clean_acc = evaluate_batch(model, test_batch.images, test_batch.labels)
    adv_loss, adv_acc = evaluate_batch(
        model, test_batch.images + pert, test_batch.labels
    )

    print("\nFinal Test Evaluation:")
    print(f"  Clean Accuracy:       {clean_acc:.1%}")
    print(f"  Adversarial Accuracy: {adv_acc:.1%}")

    # Launch GUI Visualizer on test batch
    plot_interactive_batch(model, test_batch.images, pert, test_batch.labels)



def main():
    real_training_run()

if __name__ == "__main__":
    main()