import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
# -*- coding: utf-8 -*-
"""
교사 개인 시간표 + 학생 6명 적용 (MVP)
- Streamlit 앱
- Greedy 배치(간단한 되돌리기 포함)
- CSV 템플릿 생성/다운로드, 배치 결과 시각화, CSV/ICS 내보내기

필요 패키지 (requirements.txt 예시)
streamlit
pandas
numpy

실행:
  streamlit run teacher_scheduler_app.py
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
from collections import defaultdict

st.set_page_config(page_title="교사 개인 시간표 자동 배치", layout="wide")
st.title("🗓️ 교사 개인 시간표 자동 배치 (학생 6명)")

# ------------------------------
# 기본 유틸
# ------------------------------
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAY_LABEL = {"Mon": "월", "Tue": "화", "Wed": "수", "Thu": "목", "Fri": "금"}

@st.cache_data
def sample_students_csv() -> str:
    return (
        "student_id,name,grade,homeroom,priority,service_type\n"
        "S1,김가람,2,2-1,1,국어\n"
        "S2,박나래,1,1-3,2,수학\n"
        "S3,이도현,3,3-2,2,사회성\n"
        "S4,최서윤,2,2-2,1,읽기\n"
        "S5,정민수,1,1-1,3,자립\n"
        "S6,한유진,2,2-3,2,상담\n"
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

# ------------------------------
# 사이드바: 파라미터 & 업로드
# ------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    periods_per_day = st.number_input("교시 수(일일)", min_value=4, max_value=10, value=6, step=1)
    block_minutes = st.number_input("블록 길이(분)", min_value=20, max_value=90, value=40, step=5)
    start_time = st.text_input("첫 교시 시작 시각(HH:MM)", value="08:40")
    enforce_no_consecutive = st.checkbox("연강 금지(같은 학생 연속 교시 배치 금지)", value=True)
    prefer_spread = st.slider("분산 선호 강도(높을수록 요일 분산)", 0, 10, 6)
    move_buffer = st.number_input("이동/휴식 버퍼(분, 참고용)", min_value=0, max_value=20, value=5, step=5)

    st.markdown("---")
    st.subheader("📥 데이터 업로드")

    st.markdown("학생 목록 (students.csv)")
    up_students = st.file_uploader("students.csv 업로드", type=["csv"], key="students")

    st.markdown("요구 시수 (demands.csv)")
    up_demands = st.file_uploader("demands.csv 업로드", type=["csv"], key="demands")

    st.markdown("시간 블록 (slots.csv)")
    up_slots = st.file_uploader("slots.csv 업로드", type=["csv"], key="slots")

    st.markdown("---")
    st.subheader("🧪 샘플 템플릿")
    st.download_button("students.csv 다운로드", sample_students_csv(), file_name="students.csv", mime="text/csv")
    st.download_button("demands.csv 다운로드", sample_demands_csv(), file_name="demands.csv", mime="text/csv")
    st.download_button("slots.csv 다운로드", sample_slots_csv(periods_per_day, start_time, block_minutes), file_name="slots.csv", mime="text/csv")

# ------------------------------
# 데이터 로딩
# ------------------------------
if up_students:
    students_df = pd.read_csv(up_students)
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

# 정합성 검사(간단)
need_cols_students = {"student_id","name","grade","homeroom","priority","service_type"}
need_cols_demands = {"student_id","service_type","minutes_per_week","min_block","max_blocks_per_day","preferred_days","avoid_times"}
need_cols_slots = {"day","period","start_time","end_time"}

errs = []
if not need_cols_students.issubset(students_df.columns):
    errs.append("students.csv 컬럼 부족")
if not need_cols_demands.issubset(demands_df.columns):
    errs.append("demands.csv 컬럼 부족")
if not need_cols_slots.issubset(slots_df.columns):
    errs.append("slots.csv 컬럼 부족")

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

# dict 형태로 언제든 접근 가능하게 가공
students = {
    r.student_id: dict(student_id=r.student_id, name=r.name, grade=r.grade, homeroom=r.homeroom,
                       priority=int(r.priority), service_type=r.service_type)
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

# 빠른 조회용 인덱스
slots_by_day = defaultdict(list)
for s in slots:
    slots_by_day[s["day"]].append(s)

# ------------------------------
# 배치 로직 (Greedy + 간단 Backtrack)
# ------------------------------
# 점수 함수: 선호 요일 가산, 분산 선호, 동일 학생 연속 패널티

def schedule(students:dict, demands:dict, slots:list[dict],
             enforce_no_consecutive:bool=True, prefer_spread:int=6):
    # 상태
    assigned = []  # {student_id, day, period, start, end, block_min}
    used = set()   # (day, period) 점유
    student_daily_count = defaultdict(lambda: defaultdict(int))  # student_id -> day -> blocks
    student_days_used = defaultdict(set)  # 요일 분산 점수용

    # 우선순위 높은 학생부터
    order = sorted(students.keys(), key=lambda sid: students[sid]["priority"])  # priority 낮을수록 중요하다고 가정(1=높음)

    def candidate_slots_for(sid:str):
        d = demands[sid]
        cands = []
        for s in slots:
            key = (s["day"], s["period"])
            # 이미 점유?
            if key in used: continue
            # 회피 시간?
            if key in d["avoid_times"]: continue
            cands.append(s)
        return cands

    def score_slot(sid:str, s:dict):
        d = demands[sid]
        score = 0
        # 선호 요일 가산
        if d["preferred_days"] and s["day"] in d["preferred_days"]:
            score += 5
        # 분산 선호: 새로운 요일일수록 가산
        if prefer_spread>0:
            if s["day"] not in student_days_used[sid]:
                score += prefer_spread
        # 연강 페널티(동일 학생 바로 이전 교시 배치 방지)
        if enforce_no_consecutive:
            # 같은 요일, 이전 교시가 같은 학생인지 확인
            prev_period = s["period"] - 1
            if prev_period >= 1:
                if any(a["student_id"]==sid and a["day"]==s["day"] and a["period"]==prev_period for a in assigned):
                    score -= 100  # 강한 페널티
        return score

    # 간단 되돌리기: 학생 단위로 배치 실패 시, 이전 학생의 마지막 한 블록을 해제하고 다시 시도
    def place_for_student(sid:str) -> bool:
        d = demands[sid]
        remain = d["minutes_per_week"]
        block = d["min_block"]
        # 필요한 블록 수 (올림)
        blocks_needed = int(np.ceil(remain / block))
        tries = 0
        while remain > 0 and tries < (len(slots)+50):
            tries += 1
            cands = candidate_slots_for(sid)
            if not cands:
                # 되돌리기 시도
                if not backtrack_once():
                    return False
                else:
                    continue
            # 당일 최대 블록 제한 적용
            cands = [s for s in cands if student_daily_count[sid][s["day"]] < d["max_blocks_per_day"]]
            if not cands:
                if not backtrack_once():
                    return False
                else:
                    continue
            # 점수 높은 순
            cands_sorted = sorted(cands, key=lambda s: score_slot(sid, s), reverse=True)
            picked = None
            for s in cands_sorted:
                # 연강 금지 강제 차단(점수 외 안전장치)
                if enforce_no_consecutive:
                    prev_period = s["period"]-1
                    if prev_period>=1 and any(a["student_id"]==sid and a["day"]==s["day"] and a["period"]==prev_period for a in assigned):
                        continue
                picked = s
                break
            if picked is None:
                if not backtrack_once():
                    return False
                else:
                    continue
            # 할당
            assigned.append(dict(student_id=sid, day=picked["day"], period=picked["period"], start=picked["start"], end=picked["end"], block_min=block))
            used.add((picked["day"], picked["period"]))
            student_daily_count[sid][picked["day"]] += 1
            student_days_used[sid].add(picked["day"])
            remain -= block
        return remain <= 0

    def backtrack_once() -> bool:
        # 마지막으로 배치한 한 블록을 해제하여 탐색 공간을 연다
        if not assigned:
            return False
        last = assigned.pop()
        used.discard((last["day"], last["period"]))
        student_daily_count[last["student_id"]][last["day"]] -= 1
        # 요일 사용 집합은 정확성 위해 재계산
        student_days_used.clear()
        for a in assigned:
            student_days_used[a["student_id"]].add(a["day"])
        return True

    success_map = {}
    for sid in order:
        ok = place_for_student(sid)
        success_map[sid] = ok
    return assigned, success_map

assigned, success_map = schedule(students, demands, slots,
                                 enforce_no_consecutive=enforce_no_consecutive,
                                 prefer_spread=prefer_spread)

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
        "서비스": d["service_type"],
        "필요(분/주)": total_needed,
        "배치(분/주)": total_assigned,
        "충족률(%)": round(100*total_assigned/total_needed,1) if total_needed>0 else 0.0,
        "성공": "✅" if success_map.get(sid, False) else "⚠️"
    })

st.dataframe(pd.DataFrame(sum_rows))

# 주간 그리드(교사 관점)
st.subheader("🧑‍🏫 교사 주간표")

# (day, period) -> label
grid = [["" for _ in range(periods_per_day)] for __ in range(len(DAYS))]
for a in assigned:
    r = DAYS.index(a["day"]) ; c = a["period"]-1
    name = students[a["student_id"]]["name"]
    grid[r][c] = f"{name}\n({a['start']}-{a['end']})"

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
            g[r][c] = f"{stu['service_type']}\n({a['start']}-{a['end']})"
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

st.caption("MVP: 실제 현장 제약(시험, 행사, 소집단 동시배치, 이동거리 등)은 단계적으로 확장하세요. 우선순위/가중치 튜닝으로 품질을 끌어올릴 수 있습니다.")

