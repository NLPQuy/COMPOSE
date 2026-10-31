import numpy as np

from compose.data.sampling import continual_task_splits, fewshot_episodes, task_rows


def test_task_splits_partition_classes():
    targets = np.repeat(np.arange(100), 3)
    per_task, maps = continual_task_splits(targets, 10, seed=42)
    assert sorted(per_task.reshape(-1).tolist()) == list(range(100))
    rows = task_rows("continual_train", targets, per_task, maps)
    assert sum(len(r) for r in rows) == len(targets)
    # Global labels of session t are 10t .. 10t+9.
    for t, r in enumerate(rows):
        assert set(r.labels.tolist()) == set(range(10 * t, 10 * t + 10))


def test_episode_does_not_depend_on_episode_count():
    targets = np.random.RandomState(0).permutation(np.repeat(np.arange(50), 25))
    short = fewshot_episodes("s", targets, n_way=10, n_shot=5, n_query=10, n_episodes=3, seed=7)
    long = fewshot_episodes("s", targets, n_way=10, n_shot=5, n_query=10, n_episodes=20, seed=7)
    for a, b in zip(short, long):
        assert np.array_equal(a.support.index, b.support.index)
        assert np.array_equal(a.query.index, b.query.index)


def test_episode_labels_are_relative_and_disjoint():
    targets = np.random.RandomState(1).permutation(np.repeat(np.arange(20), 20))
    for ep in fewshot_episodes("s", targets, n_way=5, n_shot=3, n_query=4, n_episodes=10, seed=0):
        assert ep.support.labels.tolist() == np.repeat(np.arange(5), 3).tolist()
        assert not set(ep.support.index) & set(ep.query.index)
        assert np.array_equal(targets[ep.query.index], ep.classes[ep.query.labels])
