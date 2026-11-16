# Phân tích 5 thành công + 5 thất bại — ext

Chọn theo luật cố định (xem `case_analysis_l4b`), KHÔNG chọn tay. Thành công = video đúng ở hạng 1; thất bại = hạng > 10.
Thống kê trên 4000 truy vấn mẫu: hạng 1 = 40.7%, hạng > 10 = 25.7%; nguyên nhân trong mẫu thất bại đã chẩn đoán (400): {'candidate_recall_low': 177, 'tag_missing': 215, 'missing_edge': 166, 'graph_helped': 129, 'noisy_relation': 71, 'predicate_not_in_graph': 5, 'wrong_predicate': 10}

## 1. [THÀNH CÔNG] "a character defeats another in combat as the fallen begins to explode"

- Video đúng: `video5392` — hạng CLIP **2** → hạng cuối **1**
- Top-3 trả về: `video5392`, `video3974`, `video396`
- Chẩn đoán (luật cố định): missing_edge, graph_helped
  - không có cạnh nào giữa character#n và explode#v (quan hệ 'agent_of' của truy vấn)
  - đồ thị kéo video đúng từ hạng 2 lên 1
- Phân rã điểm đồ thị của video đúng: {'visual': -0.344, 'contains': -0.0633, 'relation': -0.0129}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0098 video5392 –chứa→ hero#n
  - +0.0025 head#n –object_of→ attack#v ←chứa– video5392
  - +0.0024 drum#n –object_of→ beat#v ←chứa– video5392
  - +0.0020 man#n –object_of→ attack#v ←chứa– video5392
  - +0.0017 field#n –prep_inv[in]→ player#n ←chứa– video5392
- Số hạng PHẢN ĐỐI:
  - -0.0231 video5392 –chứa→ fight#v
  - -0.0152 video5392 –chứa→ sword#n
  - -0.0135 video5392 –chứa→ character#n

## 2. [THÀNH CÔNG] "a man talks on a stage in front of an audience"

- Video đúng: `video1531` — hạng CLIP **2** → hạng cuối **1**
- Top-3 trả về: `video1531`, `video5674`, `video2288`
- Chẩn đoán (luật cố định): graph_helped
  - đồ thị kéo video đúng từ hạng 2 lên 1
- Phân rã điểm đồ thị của video đúng: {'visual': -0.3627, 'contains': -0.0526, 'relation': 0.0002}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0299 video1531 –chứa→ paper#n
  - +0.0122 video1531 –chứa→ actor#n
  - +0.0117 face#n –prep[of]→ man#n ←chứa– video1531
  - +0.0114 video1531 –chứa→ man#n
  - +0.0097 camera#n –object_of→ speak#v ←chứa– video1531
- Số hạng PHẢN ĐỐI:
  - -0.0538 video1531 –chứa→ audience#n
  - -0.0414 video1531 –chứa→ speak#v
  - -0.0163 video1531 –chứa→ microphone#n

## 3. [THÀNH CÔNG] "a person is making a paper flower"

- Video đúng: `video9125` — hạng CLIP **8** → hạng cuối **1**
- Top-3 trả về: `video9125`, `video3871`, `video4114`
- Chẩn đoán (luật cố định): tag_missing, missing_edge, graph_helped
  - không khái niệm nào của truy vấn là nhãn của video đúng
  - không có cạnh nào giữa paper#n và make#v (quan hệ 'object_of' của truy vấn)
  - đồ thị kéo video đúng từ hạng 8 lên 1
- Phân rã điểm đồ thị của video đúng: {'visual': -0.6152, 'contains': 0.0468, 'relation': 0.0173}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0263 video9125 –chứa→ straw#n
  - +0.0097 video9125 –chứa→ stir#v
  - +0.0084 video9125 –chứa→ chopstick#n
  - +0.0055 video9125 –chứa→ skewer#n
  - +0.0053 video9125 –chứa→ stick#n
- Số hạng PHẢN ĐỐI:
  - -0.0086 video9125 –chứa→ lid#n
  - -0.0022 vegetable#n –object_of→ stir#v ←chứa– video9125
  - -0.0015 bowl#n –object_of→ stir#v ←chứa– video9125

## 4. [THÀNH CÔNG] "bbc world news is reporting on military guarding a building outside"

- Video đúng: `video7091` — hạng CLIP **1** → hạng cuối **1**
- Top-3 trả về: `video7091`, `video2150`, `video3154`
- Chẩn đoán (luật cố định): clip_already_correct
- Phân rã điểm đồ thị của video đúng: {'visual': -0.504, 'contains': 0.0103, 'relation': -0.0236}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0193 video7091 –chứa→ soldier#n
  - +0.0169 video7091 –chứa→ walk#v
  - +0.0123 video7091 –chứa→ building#n
  - +0.0054 snow#n –object_of→ walk#v ←chứa– video7091
  - +0.0044 video7091 –chứa→ hall#n
- Số hạng PHẢN ĐỐI:
  - -0.0322 video7091 –chứa→ court#n
  - -0.0121 video7091 –chứa→ enter#v
  - -0.0066 forest#n –object_of→ walk#v ←chứa– video7091

## 5. [THÀNH CÔNG] "an older woman and man are on the street talking to an interviewer"

- Video đúng: `video4110` — hạng CLIP **1** → hạng cuối **1**
- Top-3 trả về: `video4110`, `video861`, `video8946`
- Chẩn đoán (luật cố định): missing_edge, clip_already_correct
  - không có cạnh nào giữa man#n và street#n (quan hệ 'prep' của truy vấn)
- Phân rã điểm đồ thị của video đúng: {'visual': -0.6965, 'contains': -0.0886, 'relation': -0.0106}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0134 video4110 –chứa→ street#n
  - +0.0040 video4110 –chứa→ reporter#n
  - +0.0019 ride#v –agent_of_inv→ people#n ←chứa– video4110
  - +0.0017 race#v –agent_of_inv→ people#n ←chứa– video4110
  - +0.0017 costume#n –prep_inv[in]→ people#n ←chứa– video4110
- Số hạng PHẢN ĐỐI:
  - -0.0246 video4110 –chứa→ tell#v
  - -0.0188 video4110 –chứa→ smile#v
  - -0.0173 video4110 –chứa→ people#n

## 6. [THẤT BẠI] "man in blue shirt who is student is giving an interview"

- Video đúng: `video946` — hạng CLIP **426** → hạng cuối **426** (KHÔNG nằm trong top-k ứng viên CLIP ⇒ đồ thị không thể cứu)
- Top-3 trả về: `video4222`, `video5460`, `video300`
- Chẩn đoán (luật cố định; nguyên nhân thất bại): candidate_recall_low, tag_missing
  - video đúng ở hạng CLIP 426 > k = 50
  - không khái niệm nào của truy vấn là nhãn của video đúng
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - (không có số hạng đồ thị dương)
- Vì sao `video4222` (hạng 1, sai) được xếp trên:
  - +0.0077 cook#v –agent_of_inv→ man#n ←chứa– video4222
  - +0.0056 show#v –agent_of_inv→ man#n ←chứa– video4222
  - +0.0050 costume#n –prep_inv[in]→ people#n ←chứa– video4222
  - +0.0036 jump#v –agent_of_inv→ guy#n ←chứa– video4222
  - +0.0034 cell#n –object_of→ speak#v ←chứa– video4222

## 7. [THẤT BẠI] "a man is talking about working with metal"

- Video đúng: `video6297` — hạng CLIP **49** → hạng cuối **25**
- Top-3 trả về: `video4222`, `video2493`, `video2169`
- Chẩn đoán (luật cố định; nguyên nhân thất bại): missing_edge, graph_helped
  - không có cạnh nào giữa man#n và work#v (quan hệ 'agent_of' của truy vấn)
  - không có cạnh nào giữa metal#n và work#v (quan hệ 'object_of' của truy vấn)
  - đồ thị kéo video đúng từ hạng 49 lên 25
- Phân rã điểm đồ thị của video đúng: {'visual': -0.5655, 'contains': 0.0319, 'relation': 0.0124}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0614 video6297 –chứa→ burn#v
  - +0.0191 video6297 –chứa→ melt#v
  - +0.0099 video6297 –chứa→ metal#n
  - +0.0080 noodle#n –object_of→ boil#v ←chứa– video6297
  - +0.0063 soup#n –object_of→ boil#v ←chứa– video6297
- Số hạng PHẢN ĐỐI:
  - -0.0418 video6297 –chứa→ chain#n
  - -0.0203 video6297 –chứa→ row#n
  - -0.0060 video6297 –chứa→ machine#n
- Vì sao `video4222` (hạng 1, sai) được xếp trên:
  - +0.0065 camera#n –object_of→ speak#v ←chứa– video4222
  - +0.0063 cook#v –agent_of_inv→ man#n ←chứa– video4222
  - +0.0054 show#v –agent_of_inv→ man#n ←chứa– video4222
  - +0.0047 jump#v –agent_of_inv→ guy#n ←chứa– video4222
  - +0.0033 cell#n –object_of→ speak#v ←chứa– video4222

## 8. [THẤT BẠI] "a guy talks to his baby while the baby lays on the bed"

- Video đúng: `video8698` — hạng CLIP **10** → hạng cuối **15**
- Top-3 trả về: `video4090`, `video7238`, `video6811`
- Chẩn đoán (luật cố định; nguyên nhân thất bại): wrong_predicate, missing_edge, noisy_relation
  - cạnh baby#n–svo–bed#n mang vị từ ['lie in', 'lie on', 'play on'], không có 'lay on' của truy vấn
  - không có cạnh nào giữa guy#n và lay#v (quan hệ 'agent_of' của truy vấn)
  - không có cạnh nào giữa guy#n và baby#n (quan hệ 'prep' của truy vấn)
  - tổng đóng góp quan hệ cho video đúng âm (-0.0487)
- Phân rã điểm đồ thị của video đúng: {'visual': -0.7777, 'contains': -0.0982, 'relation': -0.0487}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0186 video8698 –chứa→ sang#n
  - +0.0060 video8698 –chứa→ child#n
  - +0.0026 studio#n –object_of→ sing#v ←chứa– video8698
  - +0.0026 mike#n –object_of→ sing#v ←chứa– video8698
  - +0.0021 stand#v –agent_of_inv→ child#n ←chứa– video8698
- Số hạng PHẢN ĐỐI:
  - -0.0320 video8698 –chứa→ laugh#v
  - -0.0212 video8698 –chứa→ baby#n
  - -0.0206 video8698 –chứa→ toddler#n
- Vì sao `video4090` (hạng 1, sai) được xếp trên:
  - +0.0143 video4090 –chứa→ face#n
  - +0.0060 video4090 –chứa→ child#n
  - +0.0047 powder#n –prep[on]→ face#n ←chứa– video4090
  - +0.0044 video4090 –chứa→ wake#v
  - +0.0032 boat#n –object_of→ lay#v ←chứa– video4090

## 9. [THẤT BẠI] "a child performing on the voice"

- Video đúng: `video3598` — hạng CLIP **10** → hạng cuối **11**
- Top-3 trả về: `video5290`, `video9212`, `video5340`
- Chẩn đoán (luật cố định; nguyên nhân thất bại): tag_missing, missing_edge, noisy_relation
  - không khái niệm nào của truy vấn là nhãn của video đúng
  - không có cạnh nào giữa child#n và perform#v (quan hệ 'agent_of' của truy vấn)
  - tổng đóng góp quan hệ cho video đúng âm (-0.0236)
- Phân rã điểm đồ thị của video đúng: {'visual': -0.4797, 'contains': -0.1268, 'relation': -0.0236}
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - +0.0141 video3598 –chứa→ sang#n
  - +0.0065 video3598 –chứa→ sing#v
  - +0.0047 mike#n –object_of→ sing#v ←chứa– video3598
  - +0.0041 studio#n –object_of→ sing#v ←chứa– video3598
  - +0.0034 guitar#n –object_of→ sing#v ←chứa– video3598
- Số hạng PHẢN ĐỐI:
  - -0.0487 video3598 –chứa→ kid#n
  - -0.0381 video3598 –chứa→ shout#v
  - -0.0358 video3598 –chứa→ yell#v
- Vì sao `video5290` (hạng 1, sai) được xếp trên:
  - +0.0202 video5290 –chứa→ studio#n
  - +0.0141 video5290 –chứa→ sang#n
  - +0.0065 video5290 –chứa→ sing#v
  - +0.0052 piano#n –object_of→ perform#v ←chứa– video5290
  - +0.0047 mike#n –object_of→ sing#v ←chứa– video5290

## 10. [THẤT BẠI] "a man is talking while playing his video game"

- Video đúng: `video7199` — hạng CLIP **340** → hạng cuối **340** (KHÔNG nằm trong top-k ứng viên CLIP ⇒ đồ thị không thể cứu)
- Top-3 trả về: `video8705`, `video1171`, `video7673`
- Chẩn đoán (luật cố định; nguyên nhân thất bại): candidate_recall_low, tag_missing
  - video đúng ở hạng CLIP 340 > k = 50
  - không khái niệm nào của truy vấn là nhãn của video đúng
- Đường đi / quan hệ ỦNG HỘ video đúng (đóng góp thật vào điểm):
  - (không có số hạng đồ thị dương)
- Vì sao `video8705` (hạng 1, sai) được xếp trên:
  - +0.0228 video8705 –chứa→ say#v
  - +0.0070 camera#n –object_of→ say#v ←chứa– video8705
  - +0.0070 camera#n –object_of→ speak#v ←chứa– video8705
  - +0.0053 video8705 –chứa→ guy#n
  - +0.0051 cell#n –object_of→ speak#v ←chứa– video8705
