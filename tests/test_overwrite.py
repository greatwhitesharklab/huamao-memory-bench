"""W108 写入覆盖＋审计日志单测（stub 实例，不加载模型）。
主键＝(entity, attribute) 结构化字段；默认关＝原追加行为；覆盖开＝原地顶替＋superseded 审计。"""
import unittest
from huamao_bkv import BKVMemory


def bare(overwrite=False):
    """绕过 __init__（不加载模型），只装配落行逻辑所需状态。"""
    b = BKVMemory.__new__(BKVMemory)
    b.cache, b.keys, b.meta, b.superseded = [], [], [], []
    b.overwrite_conflicts = overwrite
    b.Kdoc = b.msk = None
    b.state = {'extra': None, 'kv': {}}
    return b


class TestOverwriteSlot(unittest.TestCase):
    def test_default_off_never_overwrites(self):
        b = bare(overwrite=False)
        b._store_row('israel', 'founded by', 'Ben-Gurion', 'row1', 'k1')
        self.assertIsNone(b._overwrite_slot('israel', 'founded by'))

    def test_same_key_found(self):
        b = bare(overwrite=True)
        b._store_row('israel', 'founded by', 'Ben-Gurion', 'row1', 'k1')
        self.assertEqual(b._overwrite_slot('israel', 'founded by'), 0)

    def test_different_attribute_not_conflict(self):
        # Israel 两关系：同实体不同 attribute 不得撞键
        b = bare(overwrite=True)
        b._store_row('israel', 'founded by', 'Ben-Gurion', 'row1', 'k1')
        b._store_row('israel', 'located in continent of', 'Asia', 'row2', 'k2')
        self.assertEqual(len(b.meta), 2)
        self.assertIsNone(b._overwrite_slot('israel', 'other rel'))

    def test_different_entity_not_conflict(self):
        b = bare(overwrite=True)
        b._store_row('goaltender', 'sport of', 'ice hockey', 'row1', 'k1')
        self.assertIsNone(b._overwrite_slot('placekicker', 'sport of'))


class TestStoreRow(unittest.TestCase):
    def test_off_appends_duplicates(self):
        b = bare(overwrite=False)
        b._store_row('goaltender', 'sport of', 'ice hockey', 'row1', 'k1')
        b._store_row('goaltender', 'sport of', 'pesäpallo', 'row2', 'k2')
        self.assertEqual(len(b.meta), 2)          # 原追加行为：冲突双方共存
        self.assertEqual(b.superseded, [])

    def test_on_replaces_in_place(self):
        b = bare(overwrite=True)
        b._store_row('goaltender', 'sport of', 'ice hockey', 'row1', 'k1')
        slot = b._store_row('goaltender', 'sport of', 'pesäpallo', 'row2', 'k2')
        self.assertEqual(slot, 0)
        self.assertEqual(len(b.meta), 1)          # 行数不变，原地顶替
        self.assertEqual(b.meta[0], ('goaltender', 'sport of', 'pesäpallo'))
        self.assertEqual(b.cache[0], 'row2')
        self.assertEqual(b.keys[0], 'k2')

    def test_audit_log_keeps_old_row(self):
        b = bare(overwrite=True)
        b._store_row('goaltender', 'sport of', 'ice hockey', 'row1', 'k1')
        b._store_row('goaltender', 'sport of', 'pesäpallo', 'row2', 'k2')
        self.assertEqual(len(b.superseded), 1)
        rec = b.superseded[0]
        self.assertEqual(rec['key'], ('goaltender', 'sport of'))
        self.assertEqual(rec['old_meta'], ('goaltender', 'sport of', 'ice hockey'))
        self.assertEqual(rec['old_cache'], 'row1')   # 旧 K/V 可回滚
        self.assertEqual(rec['old_keys'], 'k1')
        self.assertEqual(rec['new_meta'], ('goaltender', 'sport of', 'pesäpallo'))

    def test_other_rows_not_shifted(self):
        b = bare(overwrite=True)
        b._store_row('a', 'r', 'v1', 'rowA', 'kA')
        b._store_row('b', 'r', 'v1', 'rowB', 'kB')
        b._store_row('a', 'r', 'v2', 'rowA2', 'kA2')
        self.assertEqual(b.meta[1], ('b', 'r', 'v1'))   # 其余行行号不动
        self.assertEqual(b.cache[1], 'rowB')

    def test_latest_wins_chain(self):
        b = bare(overwrite=True)
        for i, v in enumerate(['v1', 'v2', 'v3']):
            b._store_row('e', 'r', v, f'row{i}', f'k{i}')
        self.assertEqual(len(b.meta), 1)
        self.assertEqual(b.meta[0][2], 'v3')
        self.assertEqual(len(b.superseded), 2)


class TestReset(unittest.TestCase):
    def test_reset_clears_audit(self):
        b = bare(overwrite=True)
        b._store_row('e', 'r', 'v1', 'row1', 'k1')
        b._store_row('e', 'r', 'v2', 'row2', 'k2')
        b.reset()
        self.assertEqual(b.meta, [])
        self.assertEqual(b.superseded, [])


if __name__ == '__main__':
    unittest.main()