
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
from collections import defaultdict
import itertools
# -*- coding: utf-8 -*-
"""
교사 개인 시간표 + 학생 6명 적용 (MVP+)
- Streamlit 앱
- Greedy 배치(간단한 되돌리기 & 그룹 편성)
- CSV 템플릿 생성/다운로드, 배치 결과 시각화, CSV/ICS 내보내기
- 신규 기능:
    1) 수업당 최대 인원(그룹 크기) 설정
    2) 함께 배정 금지(학생 페어 금지)
    3) 학년/성별 기준 묶음(엄격/선호)
    4) 학생 개인 시간표 업로드 시, 음악/미술/체육/기타 과목 시간 자동 회피

필요 패키지 (requirements.txt 예시)
streamlit
pandas
numpy

실행:
    streamlit run teacher_scheduler_app.py
"""

st.set_page_config(page_title="교사 개인 시간표 자동 배치", layout="wide")
st.title("🗓️ 교사 개인 시간표 자동 배치 (학생 6명)")

# ------------------------------
# 기본 유틸
# ------------------------------
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAY_LABEL = {"Mon": "월", "Tue": "화", "Wed": "수", "Thu": "목", "Fri": "금"}
BLOCKABLE_SUBJECTS_DEFAULT = ["음악", "미술", "체육"]

@st.cache_data
def sample_students_csv() -> str:
    return (
        "student_id,name,grade,homeroom,gender,priority,service_type\n"
        "S1,김가람,2,2-1,F,1,국어\n"
        "S2,박나래,1,1-3,F,2,수학\n"
        "S3,이도현,3,3-2,M,2,사회성\n"
        "S4,최서윤,2,2-2,F,1,읽기\n"
        "S5,정민수,1,1-1,M,3,자립\n"
        "S6,한유진,2,2-3,F,2,상담\n"
    )

@st.cache_data
def sample_demands_csv() -> str:
    # preferred_days: Mon|Wed 처럼 파이프 구분, 없으면 빈칸
    # avoid_times: Mon-3|Tue-2 처럼 "요일-교시" 파이프 구분, 없으면 빈칸
    return (
        "student_id,service_type,minutes_per_week,min_block,max_blocks_per_day,preferred_days,avoid_times\n"
        "S1,국어,120,40,1,Mon|Wed,\n"
        "S2,수학,90,30,2,Mon|Wed,\n"
        "S3,사회성,60,30,2,Fri,Wed-3\n"
        "S4,읽기,120,40,2,Tue|Thu,\n"
        "S5,자립,60,30,1,,Fri-5\n"
        "S6,상담,60,30,1,,Mon-2|Thu-4\n"
    )
  
@st.cache_data
def sample_slots_csv(periods_per_day:int=6, start="08:40", block_min:int=40) -> str:
    # 단순한 교시 시간 생성 (모든 요일 동일)
    base = datetime(2000,1,1,int(start.split(":")[0]), int(start.split(":")[1]))
    rows = ["day,period,start_time,end_time"]
    for d in DAYS:
        t = base
        for p in range(1, periods_per_day+1):
            s = t.strftime("%H:%M")
            e = (t + timedelta(minutes=block_min)).strftime("%H:%M")
            rows.append(f"{d},{p},{s},{e}")
            t += timedelta(minutes=block_min)
    return "\n".join(rows) + "\n"

@st.cache_data
def sample_student_timetable_csv() -> str:
    return (
        "student_id,day,period,subject\n"
        "S1,Mon,2,음악\n"
        "S1,Wed,4,수학\n"
        "S2,Mon,3,미술\n"
        "S3,Wed,3,체육\n"
        "S4,Tue,2,국어\n"
        "S5,Fri,5,체육\n"
        "S6,Thu,4,미술\n"
    )


# ------------------------------
# 사이드바: 파라미터 & 업로드
# ------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    periods_per_day = st.number_input("교시 수(일일)", min_value=4, max_value=10, value=6, step=1)
    block_minutes = st.number_input("블록 길이(분)", min_value=20, max_value=90, value=40, step=5)
    start_time = st.text_input("첫 교시 시작 시각(HH:MM)", value="08:40")

    st.markdown("---")
    st.subheader("👥 그룹/묶음 옵션")
    max_group_size = st.number_input("수업당 최대 인원", min_value=1, max_value=6, value=2, step=1)
    grouping_key = st.selectbox("묶음 기준", options=["없음", "학년", "성별"], index=0)
    grouping_mode_strict = st.checkbox("엄격 적용(모두 동일해야 편성)", value=False, help="해제 시 동일 기준이면 가산점만 부여")

    st.markdown("---")
    st.subheader("🚫 함께 배정 금지 페어")
    incompatible_input = st.text_input("쉼표로 학생ID 쌍 입력 (예: S1-S3, S2-S5)")

    st.markdown("---")
    st.subheader("📥 데이터 업로드")

    st.markdown("학생 목록 (students.csv)")
    up_students = st.file_uploader("students.csv 업로드", type=["csv"], key="students")

    st.markdown("요구 시수 (demands.csv)")
    up_demands = st.file_uploader("demands.csv 업로드", type=["csv"], key="demands")

    st.markdown("시간 블록 (slots.csv)")
    up_slots = st.file_uploader("slots.csv 업로드", type=["csv"], key="slots")

    st.markdown("학생 개인 시간표 (student_timetable.csv)")
    up_timetable = st.file_uploader("student_timetable.csv 업로드 (선택)", type=["csv"], key="timetable")

    st.markdown("---")
    st.subheader("🖍️ 시간표 차단 과목")
    blockable_subjects = st.text_input(
        "차단 과목(쉼표 구분)", value=", ".join(BLOCKABLE_SUBJECTS_DEFAULT)
    )

    st.markdown("---")
    st.subheader("🧪 샘플 템플릿")
    st.download_button("students.csv 다운로드", sample_students_csv(), file_name="students.csv", mime="text/csv")
    st.download_button("demands.csv 다운로드", sample_demands_csv(), file_name="demands.csv", mime="text/csv")
    st.download_button("slots.csv 다운로드", sample_slots_csv(periods_per_day, start_time, block_minutes), file_name="slots.csv", mime="text/csv")
    st.download_button("student_timetable.csv 다운로드", sample_student_timetable_csv(), file_name="student_timetable.csv", mime="text/csv")

# ------------------------------
# 데이터 로딩
# ------------------------------

if up_students:
    students_df = pd.read_csv(up_students)
    # 컬럼명 체크 및 fallback
    expected_cols = ["student_id","name","grade","homeroom","gender","priority","service_type"]
    if not all(col in students_df.columns for col in expected_cols):
        st.warning("students.csv의 컬럼명이 올바르지 않습니다. 샘플 데이터를 대신 사용합니다.")
        students_df = pd.read_csv(StringIO(sample_students_csv()))
else:
    students_df = pd.read_csv(StringIO(sample_students_csv()))

if up_demands:
    demands_df = pd.read_csv(up_demands)
else:
    demands_df = pd.read_csv(StringIO(sample_demands_csv()))

if up_slots:
    slots_df = pd.read_csv(up_slots)
else:
    slots_df = pd.read_csv(StringIO(sample_slots_csv(periods_per_day, start_time, block_minutes)))

if up_timetable:
    timetable_df = pd.read_csv(up_timetable)
else:
    timetable_df = pd.read_csv(StringIO(sample_student_timetable_csv()))

# 정합성 검사(간단)
need_cols_students = {"student_id","name","grade","homeroom","gender","priority","service_type"}
need_cols_demands = {"student_id","service_type","minutes_per_week","min_block","max_blocks_per_day","preferred_days","avoid_times"}
need_cols_slots = {"day","period","start_time","end_time"}
need_cols_timetable = {"student_id","day","period","subject"}

errs = []
if not need_cols_students.issubset(students_df.columns):
    errs.append("students.csv 컬럼 부족 (student_id,name,grade,homeroom,gender,priority,service_type)")
if not need_cols_demands.issubset(demands_df.columns):
    errs.append("demands.csv 컬럼 부족")
if not need_cols_slots.issubset(slots_df.columns):
    errs.append("slots.csv 컬럼 부족")
if not need_cols_timetable.issubset(timetable_df.columns):
    errs.append("student_timetable.csv 컬럼 부족")

if errs:
    st.error("\n".join(errs))
    st.stop()

# 문자열 파싱 보조
def parse_days(s: str) -> list[str]:
    s = str(s).strip()
    if not s or s == "nan": return []
    return [x.strip() for x in s.split("|") if x.strip()]

def parse_avoid(s: str) -> set[tuple[str,int]]:
    s = str(s).strip()
    if not s or s == "nan": return set()
    out = set()
    for token in s.split("|"):
        token = token.strip()
        if not token: continue
        try:
            d, p = token.split("-")
            out.add((d.strip(), int(p)))
        except Exception:
            pass
    return out

# incompatible pairs 파싱 ("S1-S3, S2-S5")
def parse_incompatibles(s: str) -> set[frozenset[str]]:
    pairs = set()
    if not s: return pairs
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a,b = token.split('-',1)
            a,b = a.strip(), b.strip()
            if a and b and a!=b:
                pairs.add(frozenset([a,b]))
    return pairs

incompatible_pairs = parse_incompatibles(incompatible_input)

# dict 형태로 언제든 접근 가능하게 가공
students = {
    r.student_id: dict(student_id=r.student_id, name=r.name, grade=int(r.grade), homeroom=r.homeroom,
                       gender=str(r.gender), priority=int(r.priority), service_type=r.service_type)
    for r in students_df.itertuples(index=False)
}

# 동일 학생 multi-row 가능(복수 서비스형태), 여기서는 1행 기준 처리(학생 6명 MVP)
demands = {}
for r in demands_df.itertuples(index=False):
    demands[r.student_id] = dict(
        student_id=r.student_id,
        service_type=r.service_type,
        minutes_per_week=int(r.minutes_per_week),
        min_block=int(r.min_block),
        max_blocks_per_day=int(r.max_blocks_per_day),
        preferred_days=parse_days(r.preferred_days),
        avoid_times=parse_avoid(r.avoid_times)
    )

# 슬롯 목록
slots = []
for r in slots_df.itertuples(index=False):
    if r.day not in DAYS: # 잘못된 요일은 스킵
        continue
    slots.append(dict(day=r.day, period=int(r.period), start=str(r.start_time), end=str(r.end_time)))

# 교시 정렬
slots = sorted(slots, key=lambda x: (DAYS.index(x["day"]), x["period"]))

# 학생 개인 시간표 -> 차단 슬롯 구축
block_subjects = [s.strip() for s in str(blockable_subjects).split(',') if s.strip()]
forbidden_slots_by_student: dict[str, set[tuple[str,int]]] = defaultdict(set)
for r in timetable_df.itertuples(index=False):
    sid = str(r.student_id)
    subj = str(r.subject)
    if r.day in DAYS and subj:
        if any(b in subj for b in block_subjects):
            try:
                period = int(r.period)
                forbidden_slots_by_student[sid].add((r.day, period))
            except Exception:
                pass

# 빠른 조회용 인덱스
slots_by_day = defaultdict(list)
for s in slots:
    slots_by_day[s["day"]].append(s)

# ------------------------------
# 배치 로직 (그룹 편성 지원)
# ------------------------------
# - 한 슬롯에 최대 max_group_size명까지 배치 가능(교사 1명)
# - 함께 배정 금지 페어 미충돌
# - grouping_key(학년/성별) 기준 엄격/선호 반영
# - 학생 개인 시간표에서 차단 과목 시간 회피 + 기존 avoid_times 준수


def schedule(students:dict, demands:dict, slots:list[dict],
             max_group_size:int=2,
             grouping_key:str="없음",
             grouping_mode_strict:bool=False,
             enforce_no_consecutive:bool=True,
             prefer_spread:int=6):
    # 상태
    assigned = []  # {student_id, day, period, start, end, block_min}
    used_count = defaultdict(int)   # (day, period) -> 사용 인원
    student_daily_count = defaultdict(lambda: defaultdict(int))  # student_id -> day -> blocks
    student_days_used = defaultdict(set)  # 요일 분산 점수용

    # 우선순위 높은 학생부터
    order = sorted(students.keys(), key=lambda sid: students[sid]["priority"])  # priority 낮을수록 중요

    def is_incompatible(a:str, b:str) -> bool:
        return frozenset([a,b]) in incompatible_pairs

    def available_for(sid:str, s:dict) -> bool:
        d = demands[sid]
        key = (s["day"], s["period"])
        if key in d["avoid_times"]:
            return False
        if key in forbidden_slots_by_student.get(sid, set()):
            return False
        if student_daily_count[sid][s["day"]] >= d["max_blocks_per_day"]:
            return False
        # 연강 금지
        if enforce_no_consecutive:
            prev_period = s["period"] - 1
            if prev_period >= 1:
                if any(ae["student_id"]==sid and ae["day"]==s["day"] and ae["period"]==prev_period for ae in assigned):
                    return False
        return True

    def group_compat_ok(group:list[str]) -> bool:
        # 페어 금지 위배?
        for a,b in itertools.combinations(group, 2):
            if is_incompatible(a,b):
                return False
        if grouping_key == "없음":
            return True
        if grouping_mode_strict:
            if grouping_key == "학년":
                vals = {students[s]["grade"] for s in group}
            else: # 성별
                vals = {students[s]["gender"] for s in group}
            return len(vals) == 1
        return True  # 선호 모드는 점수에서 반영

    def group_score_bonus(group:list[str]) -> int:
        if grouping_key == "없음" or grouping_mode_strict:
            return 0
        if grouping_key == "학년":
            same = len({students[s]["grade"] for s in group})==1
        else:
            same = len({students[s]["gender"] for s in group})==1
        return 3 if same else 0

    def score_slot_for_sid(sid:str, s:dict) -> int:
        d = demands[sid]
        score = 0
        # 선호 요일 가산
        if d["preferred_days"] and s["day"] in d["preferred_days"]:
            score += 5
        # 분산 선호: 새로운 요일일수록 가산
        if prefer_spread>0 and s["day"] not in student_days_used[sid]:
            score += prefer_spread
        return score

    def pick_group_for_slot(primary_sid:str, s:dict) -> list[str]:
        # 기본은 주 학생 + (호출 가능 학생 중) 호환되는 학생들로 채우기
        group = [primary_sid]
        if max_group_size == 1:
            return group
        # 후보: 아직 시간이 남아 있고, 이 슬롯을 사용할 수 있는 학생들
        remaining_slots = max_group_size - 1
        # 다른 학생들 후보 정렬: 점수 높은 순(해당 슬롯 선호/분산)
        candidates = []
        for sid in order:
            if sid == primary_sid:
                continue
            # 이미 같은 슬롯에 있는지(중복 방지)
            if any(ae["student_id"]==sid and ae["day"]==s["day"] and ae["period"]==s["period"] for ae in assigned):
                continue
            d = demands[sid]
            remain = d["minutes_per_week"] - sum(ae["block_min"] for ae in assigned if ae["student_id"]==sid)
            if remain <= 0:
                continue
            if not available_for(sid, s):
                continue
            candidates.append((score_slot_for_sid(sid, s), sid))
        candidates.sort(reverse=True)
        for _score, sid in candidates:
            trial = group + [sid]
            if not group_compat_ok(trial):
                continue
            group = trial
            remaining_slots -= 1
            if remaining_slots == 0:
                break
        return group

    def place_block_group(sids:list[str], s:dict, block:int):
        # 그룹 전체 할당
        for sid in sids:
            assigned.append(dict(student_id=sid, day=s["day"], period=s["period"], start=s["start"], end=s["end"], block_min=block))
            student_daily_count[sid][s["day"]] += 1
            student_days_used[sid].add(s["day"])
        used_count[(s["day"], s["period"])] += len(sids)

    def remove_last_group(sids:list[str], s:dict):
        # 최근에 쌓인 순서로 pop
        for _ in sids[::-1]:
            last = assigned.pop()
            student_daily_count[last["student_id"]][last["day"]] -= 1
            # days_used는 정확을 위해 재계산
        used_count[(s["day"], s["period"])] -= len(sids)
        # 재계산
        student_days_used.clear()
        for a in assigned:
            student_days_used[a["student_id"]].add(a["day"])

    def place_for_student(primary_sid:str) -> bool:
        d = demands[primary_sid]
        remain = d["minutes_per_week"] - sum(ae["block_min"] for ae in assigned if ae["student_id"]==primary_sid)
        block = d["min_block"]
        tries = 0
        while remain > 0 and tries < (len(slots)+100):
            tries += 1
            # 후보 슬롯: 용량 여유 + 본인 사용 가능
            cands = [s for s in slots if used_count[(s["day"], s["period"]) ] < max_group_size and available_for(primary_sid, s)]
            if not cands:
                return False
            # 점수 높은 슬롯 우선
            cands_sorted = sorted(cands, key=lambda s: score_slot_for_sid(primary_sid, s), reverse=True)
            placed = False
            for s in cands_sorted:
                # 그룹 구성
                group = pick_group_for_slot(primary_sid, s)
                if not group_compat_ok(group):
                    continue
                # 엄격 모드 확인(이미 group_compat_ok에서 체크), 선호 모드면 보너스만
                bonus = group_score_bonus(group)
                # 최종 배치
                place_block_group(group, s, block)
                remain -= block
                placed = True
                break
            if not placed:
                return False
        return remain <= 0

    success_map = {}
    for sid in order:
        ok = place_for_student(sid)
        success_map[sid] = ok
    return assigned, success_map

assigned, success_map = schedule(
    students, demands, slots,
    max_group_size=max_group_size,
    grouping_key=grouping_key,
    grouping_mode_strict=grouping_mode_strict,
    enforce_no_consecutive=True,
    prefer_spread=6,
)

# ------------------------------
# 결과 테이블/그리드
# ------------------------------
st.subheader("📊 배치 결과 요약")

sum_rows = []
for sid, stu in students.items():
    d = demands[sid]
    total_needed = d["minutes_per_week"]
    total_assigned = sum(a["block_min"] for a in assigned if a["student_id"]==sid)
    sum_rows.append({
        "학생": f"{stu['name']} ({sid})",
        "학년": stu["grade"],
        "성별": stu["gender"],
        "서비스": d["service_type"],
        "필요(분/주)": total_needed,
        "배치(분/주)": total_assigned,
        "충족률(%)": round(100*total_assigned/total_needed,1) if total_needed>0 else 0.0,
        "성공": "✅" if success_map.get(sid, False) else "⚠️"
    })

st.dataframe(pd.DataFrame(sum_rows))

# 주간 그리드(교사 관점)
st.subheader("🧑‍🏫 교사 주간표")

# (day, period) -> label 리스트(그룹 표기)
labels = defaultdict(list)
for a in assigned:
    labels[(a["day"], a["period"])].append(students[a["student_id"]]["name"])

grid = [["" for _ in range(periods_per_day)] for __ in range(len(DAYS))]
for s in slots:
    r = DAYS.index(s["day"]) ; c = s["period"]-1
    names = labels.get((s["day"], s["period"]))
    if names:
        grid[r][c] = ", ".join(names) + f"\n({s['start']}-{s['end']})"

teacher_df = pd.DataFrame(grid, index=[DAY_LABEL[d] for d in DAYS], columns=[f"{i}교시" for i in range(1, periods_per_day+1)])
st.dataframe(teacher_df, use_container_width=True)

# 학생별 그리드 탭
st.subheader("👩‍🎓 학생별 주간표")
student_tabs = st.tabs([f"{v['name']}({k})" for k,v in students.items()])
for tab, (sid, stu) in zip(student_tabs, students.items()):
    with tab:
        g = [["" for _ in range(periods_per_day)] for __ in range(len(DAYS))]
        for a in assigned:
            if a["student_id"]!=sid: continue
            r = DAYS.index(a["day"]) ; c = a["period"]-1
            g[r][c] = f"{demands[sid]['service_type']}\n({a['start']}-{a['end']})"
        df = pd.DataFrame(g, index=[DAY_LABEL[d] for d in DAYS], columns=[f"{i}교시" for i in range(1, periods_per_day+1)])
        st.dataframe(df, use_container_width=True)

# ------------------------------
# 내보내기 (CSV / ICS)
# ------------------------------
st.subheader("⬇️ 내보내기")

def make_assignments_csv(assigned:list[dict]) -> str:
    rows = ["student_id,student_name,day,period,start,end,minutes"]
    for a in assigned:
        rows.append(
            f"{a['student_id']},{students[a['student_id']]['name']},{a['day']},{a['period']},{a['start']},{a['end']},{a['block_min']}"
        )
    return "\n".join(rows)

csv_data = make_assignments_csv(assigned)
st.download_button("배치 결과 CSV 다운로드", csv_data, file_name="assignments.csv", mime="text/csv")

# 간단 ICS 생성(주차 기준 날짜 매핑: 임의로 2025-01-06(월) 주)
BASE_MONDAY = datetime(2025,1,6)
weekday_idx = {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4}

def to_dt(base_monday:datetime, day:str, hm:str) -> datetime:
    d = base_monday + timedelta(days=weekday_idx[day])
    h,m = map(int, hm.split(":"))
    return d.replace(hour=h, minute=m, second=0, microsecond=0)

def dtstamp(dt:datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")

def make_ics(assigned:list[dict]) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//TeacherScheduler//MVP//KR"
    ]
    now = datetime.utcnow()
    for a in assigned:
        start_dt = to_dt(BASE_MONDAY, a["day"], a["start"])  # 해당 주 기준 1회 일정
        end_dt = to_dt(BASE_MONDAY, a["day"], a["end"]) 
        summary = f"{students[a['student_id']]['name']} 수업"
        desc = f"서비스: {demands[a['student_id']]['service_type']} / 블록: {a['block_min']}분"
        uid = f"{a['student_id']}-{a['day']}-{a['period']}@teacherscheduler"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp(now)}Z",
            f"DTSTART:{dtstamp(start_dt)}",
            f"DTEND:{dtstamp(end_dt)}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{desc}",
            "END:VEVENT"
        ]
    lines.append("END:VCALENDAR")
    return "\n".join(lines)

ics_data = make_ics(assigned)
st.download_button("ICS(캘린더) 다운로드", ics_data, file_name="teacher_schedule.ics", mime="text/calendar")

# ------------------------------
# 진단/로그
# ------------------------------
with st.expander("🔍 배치 로그/디버깅"):
    st.write("성공 맵:", success_map)
    st.write("할당 수:", len(assigned))
    st.dataframe(pd.DataFrame(assigned))

st.caption("MVP+: 그룹 편성/금지 페어/차단 과목 회피를 반영합니다. 소집단 고정조 편성, 이동거리 최소화, 주차별 변동 규칙 등은 추가 확장 가능합니다.")
