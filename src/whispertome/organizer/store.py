from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class OrganizerPaths:
    root: Path

    @property
    def db_file(self) -> Path:
        return self.root / "organizer.sqlite"

    @property
    def legacy_store_file(self) -> Path:
        return self.root / "store.json"


class OrganizerStore:
    def __init__(self, path: Path) -> None:
        self._paths = OrganizerPaths(path)
        self._lock = RLock()
        with self._lock:
            self._paths.root.mkdir(parents=True, exist_ok=True)
            with self._connection() as conn:
                self._init_schema(conn)
                self._migrate_legacy_json(conn)

    def add_note(
        self,
        *,
        content: str,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> JsonObject:
        content = content.strip()
        if not content:
            raise ValueError("note content is required")
        note = {
            "id": _new_id("note"),
            "created_at": _now_iso(),
            "title": _clean_optional(title) or _compact_title(content),
            "content": content,
            "tags": _clean_tags(tags),
        }
        with self._lock, self._connection() as conn:
            self._insert_note(conn, note)
        return note

    def list_notes(
        self,
        *,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        needle = (_clean_optional(query) or "").casefold()
        tag_filter = {tag.casefold() for tag in _clean_tags(tags)}
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY created_at DESC LIMIT ?",
                (max(1, limit * 4),),
            ).fetchall()

        notes: list[JsonObject] = []
        for row in rows:
            note = _decode_json_fields(dict(row), "tags")
            text = f"{note.get('title', '')} {note.get('content', '')}".casefold()
            note_tags = {str(tag).casefold() for tag in note.get("tags", [])}
            if needle and needle not in text:
                continue
            if tag_filter and not tag_filter.issubset(note_tags):
                continue
            notes.append(note)
            if len(notes) >= max(1, limit):
                break
        return notes

    def add_reminder(
        self,
        *,
        title: str,
        due_date: str | None = None,
        due_time: str | None = None,
        notes: str | None = None,
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> JsonObject:
        reminder = {
            "id": _new_id("rem"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "title": _required_text(title, "reminder title"),
            "due_date": _clean_optional(due_date),
            "due_time": _clean_optional(due_time),
            "notes": _clean_optional(notes),
            "priority": _clean_optional(priority) or "normal",
            "status": "pending",
            "completed_at": None,
            "tags": _clean_tags(tags),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO reminders (
                    id, created_at, updated_at, title, due_date, due_time,
                    notes, priority, status, completed_at, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(
                    reminder,
                    "id",
                    "created_at",
                    "updated_at",
                    "title",
                    "due_date",
                    "due_time",
                    "notes",
                    "priority",
                    "status",
                    "completed_at",
                    "tags",
                ),
            )
        return reminder

    def list_reminders(
        self,
        *,
        status: str | None = None,
        due_before: str | None = None,
        include_completed: bool = False,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif not include_completed:
            clauses.append("status != 'done'")
        if due_before:
            clauses.append("(due_date IS NOT NULL AND due_date <= ?)")
            params.append(due_before)
        sql = "SELECT * FROM reminders"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY due_date IS NULL, due_date, due_time IS NULL, due_time, created_at DESC"
        sql += " LIMIT ?"
        params.append(max(1, limit * 4))

        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return _filter_text_rows(rows, ("title", "notes"), needle, limit)

    def complete_reminder(
        self,
        *,
        reminder_id: str | None = None,
        title_query: str | None = None,
    ) -> JsonObject:
        with self._lock, self._connection() as conn:
            reminder = self._find_by_id_or_query(
                conn,
                table="reminders",
                item_id=reminder_id,
                query=title_query,
                query_column="title",
                open_status=True,
            )
            now = _now_iso()
            conn.execute(
                """
                UPDATE reminders
                SET status = 'done', completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, reminder["id"]),
            )
            reminder.update({"status": "done", "completed_at": now, "updated_at": now})
        return reminder

    def create_checklist(
        self,
        *,
        title: str,
        items: list[str],
        tags: list[str] | None = None,
    ) -> JsonObject:
        cleaned_items = [item.strip() for item in items if item.strip()]
        if not cleaned_items:
            raise ValueError("at least one checklist item is required")
        checklist = {
            "id": _new_id("list"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "title": _required_text(title, "checklist title"),
            "tags": _clean_tags(tags),
            "items": [_checklist_item(text) for text in cleaned_items],
        }
        with self._lock, self._connection() as conn:
            self._insert_checklist(conn, checklist)
        return checklist

    def append_checklist_items(
        self,
        *,
        checklist_id: str | None = None,
        title_query: str | None = None,
        items: list[str],
    ) -> JsonObject:
        cleaned_items = [item.strip() for item in items if item.strip()]
        if not cleaned_items:
            raise ValueError("at least one checklist item is required")
        new_items = [_checklist_item(text) for text in cleaned_items]
        with self._lock, self._connection() as conn:
            checklist = self._find_checklist(
                conn,
                checklist_id=checklist_id,
                title_query=title_query,
            )
            for item in new_items:
                self._insert_checklist_item(conn, checklist["id"], item)
            conn.execute(
                "UPDATE checklists SET updated_at = ? WHERE id = ?",
                (_now_iso(), checklist["id"]),
            )
            checklist = self._load_checklist(conn, checklist["id"], include_completed=True)
        return {"checklist": checklist, "added_items": new_items}

    def list_checklists(
        self,
        *,
        include_completed: bool = True,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM checklists ORDER BY created_at DESC LIMIT ?",
                (max(1, limit * 4),),
            ).fetchall()
            checklists = [
                self._load_checklist(conn, row["id"], include_completed=include_completed)
                for row in rows
            ]
        filtered: list[JsonObject] = []
        for checklist in checklists:
            if needle and needle not in str(checklist.get("title", "")).casefold():
                continue
            filtered.append(checklist)
            if len(filtered) >= max(1, limit):
                break
        return filtered

    def complete_checklist_item(
        self,
        *,
        checklist_id: str | None = None,
        title_query: str | None = None,
        item_id: str | None = None,
        item_text: str | None = None,
    ) -> JsonObject:
        with self._lock, self._connection() as conn:
            checklist = self._find_checklist(
                conn,
                checklist_id=checklist_id,
                title_query=title_query,
            )
            item = self._find_checklist_item(
                conn,
                checklist["id"],
                item_id=item_id,
                item_text=item_text,
            )
            now = _now_iso()
            conn.execute(
                """
                UPDATE checklist_items
                SET done = 1, completed_at = ?
                WHERE id = ?
                """,
                (now, item["id"]),
            )
            conn.execute(
                "UPDATE checklists SET updated_at = ? WHERE id = ?",
                (now, checklist["id"]),
            )
            item.update({"done": True, "completed_at": now})
            checklist = self._load_checklist(conn, checklist["id"], include_completed=True)
        return {"checklist": checklist, "item": item}

    def add_itinerary_item(
        self,
        *,
        title: str,
        date: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        location: str | None = None,
        notes: str | None = None,
        category: str | None = None,
    ) -> JsonObject:
        item = {
            "id": _new_id("event"),
            "created_at": _now_iso(),
            "title": _required_text(title, "itinerary title"),
            "date": _clean_optional(date),
            "start_time": _clean_optional(start_time),
            "end_time": _clean_optional(end_time),
            "location": _clean_optional(location),
            "notes": _clean_optional(notes),
            "category": _clean_optional(category),
        }
        with self._lock, self._connection() as conn:
            self._insert_itinerary_item(conn, item)
        return item

    def list_itinerary(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        clauses: list[str] = []
        params: list[Any] = []
        if date_from:
            clauses.append("(date IS NOT NULL AND date >= ?)")
            params.append(date_from)
        if date_to:
            clauses.append("(date IS NOT NULL AND date <= ?)")
            params.append(date_to)
        sql = "SELECT * FROM itinerary"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY date IS NULL, date, start_time IS NULL, start_time, created_at"
        sql += " LIMIT ?"
        params.append(max(1, limit * 4))

        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return _filter_text_rows(rows, ("title", "location", "notes", "category"), needle, limit)

    def add_task(
        self,
        *,
        title: str,
        notes: str | None = None,
        due_date: str | None = None,
        due_time: str | None = None,
        priority: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> JsonObject:
        task = {
            "id": _new_id("task"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "title": _required_text(title, "task title"),
            "notes": _clean_optional(notes),
            "status": "open",
            "due_date": _clean_optional(due_date),
            "due_time": _clean_optional(due_time),
            "priority": _clean_optional(priority) or "normal",
            "project_id": _clean_optional(project_id),
            "completed_at": None,
            "tags": _clean_tags(tags),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, created_at, updated_at, title, notes, status, due_date,
                    due_time, priority, project_id, completed_at, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(
                    task,
                    "id",
                    "created_at",
                    "updated_at",
                    "title",
                    "notes",
                    "status",
                    "due_date",
                    "due_time",
                    "priority",
                    "project_id",
                    "completed_at",
                    "tags",
                ),
            )
        return task

    def list_tasks(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        due_before: str | None = None,
        include_completed: bool = False,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif not include_completed:
            clauses.append("status != 'done'")
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if due_before:
            clauses.append("(due_date IS NOT NULL AND due_date <= ?)")
            params.append(due_before)
        sql = "SELECT * FROM tasks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY due_date IS NULL, due_date, due_time IS NULL, due_time"
        sql += ", created_at DESC LIMIT ?"
        params.append(max(1, limit * 4))

        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return _filter_text_rows(rows, ("title", "notes", "priority"), needle, limit)

    def complete_task(
        self,
        *,
        task_id: str | None = None,
        title_query: str | None = None,
    ) -> JsonObject:
        with self._lock, self._connection() as conn:
            task = self._find_by_id_or_query(
                conn,
                table="tasks",
                item_id=task_id,
                query=title_query,
                query_column="title",
                open_status=True,
            )
            now = _now_iso()
            conn.execute(
                """
                UPDATE tasks
                SET status = 'done', completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, task["id"]),
            )
            task.update({"status": "done", "completed_at": now, "updated_at": now})
        return task

    def create_project(
        self,
        *,
        name: str,
        description: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> JsonObject:
        project = {
            "id": _new_id("proj"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "name": _required_text(name, "project name"),
            "description": _clean_optional(description),
            "status": _clean_optional(status) or "active",
            "tags": _clean_tags(tags),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, created_at, updated_at, name, description, status, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(
                    project,
                    "id",
                    "created_at",
                    "updated_at",
                    "name",
                    "description",
                    "status",
                    "tags",
                ),
            )
        return project

    def list_projects(
        self,
        *,
        status: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM projects"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(max(1, limit * 4))

        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return _filter_text_rows(rows, ("name", "description", "status"), needle, limit)

    def save_daily_plan(
        self,
        *,
        date: str,
        title: str | None = None,
        focus: str | None = None,
        items: list[str] | None = None,
        notes: str | None = None,
    ) -> JsonObject:
        plan = {
            "id": _new_id("plan"),
            "created_at": _now_iso(),
            "date": _required_text(date, "daily plan date"),
            "title": _clean_optional(title) or f"Daily plan {date}",
            "focus": _clean_optional(focus),
            "items": [item.strip() for item in (items or []) if item.strip()],
            "notes": _clean_optional(notes),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_plans (
                    id, created_at, date, title, focus, items, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(
                    plan,
                    "id",
                    "created_at",
                    "date",
                    "title",
                    "focus",
                    "items",
                    "notes",
                ),
            )
        return plan

    def get_daily_plan(
        self,
        *,
        date: str,
        include_completed: bool = False,
    ) -> JsonObject:
        date = _required_text(date, "daily plan date")
        with self._lock, self._connection() as conn:
            plan_row = conn.execute(
                """
                SELECT * FROM daily_plans
                WHERE date = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (date,),
            ).fetchone()
            saved_plan = (
                _decode_json_fields(dict(plan_row), "items") if plan_row is not None else None
            )
        return {
            "ok": True,
            "date": date,
            "saved_plan": saved_plan,
            "tasks": self.list_tasks(
                due_before=date,
                include_completed=include_completed,
                limit=20,
            ),
            "reminders": self.list_reminders(
                due_before=date,
                include_completed=include_completed,
                limit=20,
            ),
            "itinerary": self.list_itinerary(date_from=date, date_to=date, limit=20),
        }

    def list_time_sensitive(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        days_ahead: int = 7,
        include_overdue: bool = True,
        include_completed: bool = False,
        important_only: bool = False,
        limit: int = 10,
    ) -> JsonObject:
        start = _clean_optional(date_from) or datetime.now().date().isoformat()
        end = _clean_optional(date_to) or _add_days(start, days_ahead)

        reminders = self._list_due_rows(
            "reminders",
            date_from=start,
            date_to=end,
            include_overdue=include_overdue,
            include_completed=include_completed,
            important_only=important_only,
            limit=limit,
        )
        tasks = self._list_due_rows(
            "tasks",
            date_from=start,
            date_to=end,
            include_overdue=include_overdue,
            include_completed=include_completed,
            important_only=important_only,
            limit=limit,
        )
        itinerary = self.list_itinerary(
            date_from=start,
            date_to=end,
            limit=limit,
        )
        return {
            "ok": True,
            "date_from": start,
            "date_to": end,
            "days_ahead": days_ahead,
            "include_overdue": include_overdue,
            "important_only": important_only,
            "reminders": reminders,
            "tasks": tasks,
            "itinerary": itinerary,
            "counts": {
                "reminders": len(reminders),
                "tasks": len(tasks),
                "itinerary": len(itinerary),
                "total": len(reminders) + len(tasks) + len(itinerary),
            },
        }

    def add_decision(
        self,
        *,
        title: str,
        decision: str,
        rationale: str | None = None,
        project_id: str | None = None,
        decided_on: str | None = None,
        tags: list[str] | None = None,
    ) -> JsonObject:
        item = {
            "id": _new_id("dec"),
            "created_at": _now_iso(),
            "title": _required_text(title, "decision title"),
            "decision": _required_text(decision, "decision"),
            "rationale": _clean_optional(rationale),
            "project_id": _clean_optional(project_id),
            "decided_on": _clean_optional(decided_on),
            "tags": _clean_tags(tags),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    id, created_at, title, decision, rationale,
                    project_id, decided_on, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(
                    item,
                    "id",
                    "created_at",
                    "title",
                    "decision",
                    "rationale",
                    "project_id",
                    "decided_on",
                    "tags",
                ),
            )
        return item

    def list_decisions(
        self,
        *,
        project_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        sql = "SELECT * FROM decisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY decided_on IS NULL, decided_on DESC, created_at DESC LIMIT ?"
        params.append(max(1, limit * 4))

        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return _filter_text_rows(rows, ("title", "decision", "rationale"), needle, limit)

    def add_person(
        self,
        *,
        name: str,
        relationship: str | None = None,
        organization: str | None = None,
        notes: str | None = None,
        follow_up_date: str | None = None,
        tags: list[str] | None = None,
    ) -> JsonObject:
        person = {
            "id": _new_id("person"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "name": _required_text(name, "person name"),
            "relationship": _clean_optional(relationship),
            "organization": _clean_optional(organization),
            "notes": _clean_optional(notes),
            "follow_up_date": _clean_optional(follow_up_date),
            "tags": _clean_tags(tags),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO people (
                    id, created_at, updated_at, name, relationship,
                    organization, notes, follow_up_date, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(
                    person,
                    "id",
                    "created_at",
                    "updated_at",
                    "name",
                    "relationship",
                    "organization",
                    "notes",
                    "follow_up_date",
                    "tags",
                ),
            )
        return person

    def add_person_note(
        self,
        *,
        person_id: str | None = None,
        name_query: str | None = None,
        note: str,
    ) -> JsonObject:
        note = _required_text(note, "person note")
        with self._lock, self._connection() as conn:
            person = self._find_by_id_or_query(
                conn,
                table="people",
                item_id=person_id,
                query=name_query,
                query_column="name",
                open_status=False,
            )
            timestamped = f"[{_now_iso()}] {note}"
            notes = person.get("notes")
            updated_notes = f"{notes}\n{timestamped}" if notes else timestamped
            now = _now_iso()
            conn.execute(
                "UPDATE people SET notes = ?, updated_at = ? WHERE id = ?",
                (updated_notes, now, person["id"]),
            )
            person.update({"notes": updated_notes, "updated_at": now})
        return person

    def list_people(
        self,
        *,
        query: str | None = None,
        limit: int = 10,
    ) -> list[JsonObject]:
        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM people ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                (max(1, limit * 4),),
            ).fetchall()
        return _filter_text_rows(
            rows,
            ("name", "relationship", "organization", "notes"),
            needle,
            limit,
        )

    def save_preference(
        self,
        *,
        key: str,
        preference: str,
        category: str | None = None,
        value: str | None = None,
        evidence: str | None = None,
        priority: int | None = None,
    ) -> JsonObject:
        normalized_key = _preference_key(key)
        compact_preference = _compact_line(preference, label="preference", max_chars=240)
        now = _now_iso()
        item = {
            "id": _new_id("pref"),
            "created_at": now,
            "updated_at": now,
            "key": normalized_key,
            "category": _clean_optional(category) or "general",
            "preference": compact_preference,
            "value": _clean_optional(value),
            "evidence": _compact_optional(evidence, max_chars=500),
            "active": True,
            "priority": _bounded_int(priority, default=50, minimum=0, maximum=100),
        }
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM preferences WHERE key = ?",
                (normalized_key,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO preferences (
                        id, created_at, updated_at, key, category, preference,
                        value, evidence, active, priority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _row_values(
                        item,
                        "id",
                        "created_at",
                        "updated_at",
                        "key",
                        "category",
                        "preference",
                        "value",
                        "evidence",
                        "active",
                        "priority",
                    ),
                )
            else:
                item["id"] = existing["id"]
                item["created_at"] = existing["created_at"]
                conn.execute(
                    """
                    UPDATE preferences
                    SET updated_at = ?, category = ?, preference = ?, value = ?,
                        evidence = ?, active = 1, priority = ?
                    WHERE key = ?
                    """,
                    (
                        now,
                        item["category"],
                        item["preference"],
                        item["value"],
                        item["evidence"],
                        item["priority"],
                        normalized_key,
                    ),
                )
        return item

    def list_preferences(
        self,
        *,
        active_only: bool = True,
        query: str | None = None,
        limit: int = 20,
    ) -> list[JsonObject]:
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("active = 1")
        sql = "SELECT * FROM preferences"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY active DESC, priority DESC, updated_at DESC LIMIT ?"
        params.append(max(1, limit * 4))

        needle = (_clean_optional(query) or "").casefold()
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        matches: list[JsonObject] = []
        for row in rows:
            item = _bool_fields(dict(row), "active")
            text = " ".join(
                str(item.get(column) or "")
                for column in ("key", "category", "preference", "value", "evidence")
            ).casefold()
            if needle and needle not in text:
                continue
            matches.append(item)
            if len(matches) >= max(1, limit):
                break
        return matches

    def preference_prompt_context(
        self,
        *,
        max_items: int = 12,
        max_chars: int = 900,
    ) -> str:
        max_items = max(1, max_items)
        max_chars = max(120, max_chars)
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """
                SELECT preference FROM preferences
                WHERE active = 1
                ORDER BY priority DESC, updated_at DESC
                LIMIT ?
                """,
                (max_items * 2,),
            ).fetchall()

        header = "User preferences:"
        lines: list[str] = []
        used_chars = len(header)
        for row in rows:
            preference = _compact_line(row["preference"], label="preference", max_chars=160)
            line = f"- {preference}"
            next_size = used_chars + 1 + len(line)
            if next_size > max_chars:
                break
            lines.append(line)
            used_chars = next_size
            if len(lines) >= max_items:
                break
        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    def summary(self) -> JsonObject:
        with self._lock, self._connection() as conn:
            counts = {
                "notes": _count(conn, "notes"),
                "reminders": _count(conn, "reminders"),
                "pending_reminders": _count(conn, "reminders", "status != 'done'"),
                "checklists": _count(conn, "checklists"),
                "open_checklist_items": _count(conn, "checklist_items", "done = 0"),
                "itinerary_items": _count(conn, "itinerary"),
                "tasks": _count(conn, "tasks"),
                "open_tasks": _count(conn, "tasks", "status != 'done'"),
                "projects": _count(conn, "projects"),
                "daily_plans": _count(conn, "daily_plans"),
                "decisions": _count(conn, "decisions"),
                "people": _count(conn, "people"),
                "preferences": _count(conn, "preferences"),
                "active_preferences": _count(conn, "preferences", "active = 1"),
            }
        return {"ok": True, **counts}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._paths.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                due_date TEXT,
                due_time TEXT,
                notes TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                completed_at TEXT,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checklists (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checklist_items (
                id TEXT PRIMARY KEY,
                checklist_id TEXT NOT NULL REFERENCES checklists(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS itinerary (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                date TEXT,
                start_time TEXT,
                end_time TEXT,
                location TEXT,
                notes TEXT,
                category TEXT
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                notes TEXT,
                status TEXT NOT NULL,
                due_date TEXT,
                due_time TEXT,
                priority TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                completed_at TEXT,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daily_plans (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                date TEXT NOT NULL,
                title TEXT NOT NULL,
                focus TEXT,
                items TEXT NOT NULL,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                decision TEXT NOT NULL,
                rationale TEXT,
                project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                decided_on TEXT,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                name TEXT NOT NULL,
                relationship TEXT,
                organization TEXT,
                notes TEXT,
                follow_up_date TEXT,
                tags TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                preference TEXT NOT NULL,
                value TEXT,
                evidence TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                priority INTEGER NOT NULL DEFAULT 50
            );
            CREATE TABLE IF NOT EXISTS scheduled_events (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                action_json TEXT NOT NULL,
                trigger TEXT NOT NULL,
                next_fire_at TEXT,
                recurrence_rule TEXT,
                status TEXT NOT NULL,
                last_fired_at TEXT,
                fire_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                lease_token TEXT,
                lease_pid INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_date, due_time);
            CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date, due_time);
            CREATE INDEX IF NOT EXISTS idx_itinerary_date ON itinerary(date, start_time);
            CREATE INDEX IF NOT EXISTS idx_daily_plans_date ON daily_plans(date);
            CREATE INDEX IF NOT EXISTS idx_preferences_active
                ON preferences(active, priority, updated_at);
            CREATE INDEX IF NOT EXISTS idx_scheduled_events_due
                ON scheduled_events(status, next_fire_at);
            """
        )

    def _migrate_legacy_json(self, conn: sqlite3.Connection) -> None:
        marker = conn.execute(
            "SELECT value FROM metadata WHERE key = 'legacy_json_migrated'"
        ).fetchone()
        if marker is not None:
            return

        path = self._paths.legacy_store_file
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for note in data.get("notes", []):
                self._insert_note(conn, note, ignore=True)
            for checklist in data.get("checklists", []):
                self._insert_checklist(conn, checklist, ignore=True)
            for item in data.get("itinerary", []):
                self._insert_itinerary_item(conn, item, ignore=True)
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("legacy_json_migrated", _now_iso()),
        )

    def _insert_note(
        self,
        conn: sqlite3.Connection,
        note: JsonObject,
        *,
        ignore: bool = False,
    ) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        conn.execute(
            f"{verb} INTO notes (id, created_at, title, content, tags) VALUES (?, ?, ?, ?, ?)",
            (
                note.get("id") or _new_id("note"),
                note.get("created_at") or _now_iso(),
                note.get("title") or _compact_title(str(note.get("content", ""))),
                note.get("content") or "",
                _json_dumps(_clean_tags(note.get("tags"))),
            ),
        )

    def _insert_checklist(
        self,
        conn: sqlite3.Connection,
        checklist: JsonObject,
        *,
        ignore: bool = False,
    ) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        conn.execute(
            f"""
            {verb} INTO checklists (id, created_at, updated_at, title, tags)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                checklist.get("id") or _new_id("list"),
                checklist.get("created_at") or _now_iso(),
                checklist.get("updated_at") or checklist.get("created_at") or _now_iso(),
                checklist.get("title") or "Checklist",
                _json_dumps(_clean_tags(checklist.get("tags"))),
            ),
        )
        for item in checklist.get("items", []):
            self._insert_checklist_item(conn, checklist["id"], item, ignore=ignore)

    def _insert_checklist_item(
        self,
        conn: sqlite3.Connection,
        checklist_id: str,
        item: JsonObject,
        *,
        ignore: bool = False,
    ) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        conn.execute(
            f"""
            {verb} INTO checklist_items (
                id, checklist_id, text, done, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("id") or _new_id("item"),
                checklist_id,
                item.get("text") or "",
                1 if bool(item.get("done", False)) else 0,
                item.get("created_at") or _now_iso(),
                item.get("completed_at"),
            ),
        )

    def _insert_itinerary_item(
        self,
        conn: sqlite3.Connection,
        item: JsonObject,
        *,
        ignore: bool = False,
    ) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        conn.execute(
            f"""
            {verb} INTO itinerary (
                id, created_at, title, date, start_time,
                end_time, location, notes, category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("id") or _new_id("event"),
                item.get("created_at") or _now_iso(),
                item.get("title") or "Itinerary item",
                item.get("date"),
                item.get("start_time"),
                item.get("end_time"),
                item.get("location"),
                item.get("notes"),
                item.get("category"),
            ),
        )

    def _find_checklist(
        self,
        conn: sqlite3.Connection,
        *,
        checklist_id: str | None,
        title_query: str | None,
    ) -> JsonObject:
        if checklist_id:
            row = conn.execute("SELECT * FROM checklists WHERE id = ?", (checklist_id,)).fetchone()
            if row is None:
                raise ValueError(f"checklist not found: {checklist_id}")
            return _decode_json_fields(dict(row), "tags")

        query = (_clean_optional(title_query) or "").casefold()
        rows = conn.execute("SELECT * FROM checklists ORDER BY created_at DESC").fetchall()
        if not query and len(rows) == 1:
            return _decode_json_fields(dict(rows[0]), "tags")
        matches = [
            _decode_json_fields(dict(row), "tags")
            for row in rows
            if query and query in str(row["title"]).casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError("checklist not found")
        raise ValueError("multiple matching checklists; use checklist_id")

    def _load_checklist(
        self,
        conn: sqlite3.Connection,
        checklist_id: str,
        *,
        include_completed: bool,
    ) -> JsonObject:
        row = conn.execute("SELECT * FROM checklists WHERE id = ?", (checklist_id,)).fetchone()
        if row is None:
            raise ValueError(f"checklist not found: {checklist_id}")
        checklist = _decode_json_fields(dict(row), "tags")
        sql = "SELECT * FROM checklist_items WHERE checklist_id = ?"
        params: list[Any] = [checklist_id]
        if not include_completed:
            sql += " AND done = 0"
        sql += " ORDER BY created_at"
        checklist["items"] = [
            _bool_fields(dict(item), "done") for item in conn.execute(sql, params).fetchall()
        ]
        return checklist

    def _find_checklist_item(
        self,
        conn: sqlite3.Connection,
        checklist_id: str,
        *,
        item_id: str | None,
        item_text: str | None,
    ) -> JsonObject:
        if item_id:
            row = conn.execute(
                "SELECT * FROM checklist_items WHERE checklist_id = ? AND id = ?",
                (checklist_id, item_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"checklist item not found: {item_id}")
            return _bool_fields(dict(row), "done")

        query = (_clean_optional(item_text) or "").casefold()
        rows = conn.execute(
            "SELECT * FROM checklist_items WHERE checklist_id = ?",
            (checklist_id,),
        ).fetchall()
        matches = [
            _bool_fields(dict(row), "done")
            for row in rows
            if query and query in str(row["text"]).casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError("checklist item not found")
        raise ValueError("multiple matching checklist items; use item_id")

    # ---- scheduled events (fired by the in-process scheduler) -------------

    _SCHED_COLUMNS = (
        "id", "created_at", "updated_at", "title", "action_json", "trigger",
        "next_fire_at", "recurrence_rule", "status", "last_fired_at",
        "fire_count", "last_error", "lease_token", "lease_pid",
    )

    def add_scheduled_event(
        self,
        *,
        title: str,
        action_json: str,
        trigger: str,
        next_fire_at: str | None,
        recurrence_rule: str | None = None,
    ) -> JsonObject:
        if trigger not in ("one_shot", "recurring"):
            raise ValueError("trigger must be 'one_shot' or 'recurring'")
        event = {
            "id": _new_id("evt"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "title": _required_text(title, "event title"),
            "action_json": action_json,
            "trigger": trigger,
            "next_fire_at": _clean_optional(next_fire_at),
            "recurrence_rule": _clean_optional(recurrence_rule),
            "status": "pending",
            "last_fired_at": None,
            "fire_count": 0,
            "last_error": None,
            "lease_token": None,
            "lease_pid": None,
        }
        placeholders = ", ".join("?" for _ in self._SCHED_COLUMNS)
        with self._lock, self._connection() as conn:
            conn.execute(
                f"INSERT INTO scheduled_events ({', '.join(self._SCHED_COLUMNS)}) "
                f"VALUES ({placeholders})",
                _row_values(event, *self._SCHED_COLUMNS),
            )
        return event

    def list_scheduled_events(
        self, *, include_terminal: bool = False, limit: int = 20
    ) -> list[JsonObject]:
        sql = "SELECT * FROM scheduled_events"
        if not include_terminal:
            sql += " WHERE status IN ('pending', 'firing', 'needs_review')"
        sql += " ORDER BY next_fire_at IS NULL, next_fire_at, created_at DESC LIMIT ?"
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, (max(1, limit),)).fetchall()
        return [dict(row) for row in rows]

    def cancel_scheduled_event(
        self, *, event_id: str | None = None, title_query: str | None = None
    ) -> JsonObject:
        with self._lock, self._connection() as conn:
            event = self._find_by_id_or_query(
                conn,
                table="scheduled_events",
                item_id=event_id,
                query=title_query,
                query_column="title",
                open_status=False,
            )
            if event.get("status") == "firing":
                raise ValueError("event is currently firing; cannot cancel")
            conn.execute(
                "UPDATE scheduled_events SET status = 'cancelled', next_fire_at = NULL, "
                "updated_at = ? WHERE id = ?",
                (_now_iso(), event["id"]),
            )
            event.update(status="cancelled", next_fire_at=None)
        return event

    def get_due_events(self, now_iso: str, *, limit: int = 20) -> list[JsonObject]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_events WHERE status = 'pending' "
                "AND next_fire_at IS NOT NULL AND next_fire_at <= ? "
                "ORDER BY next_fire_at LIMIT ?",
                (now_iso, max(1, limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def next_pending_event(self) -> JsonObject | None:
        """The soonest upcoming event (for the 'next up' display)."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_events WHERE status = 'pending' "
                "AND next_fire_at IS NOT NULL ORDER BY next_fire_at LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def peek_next_fire_at(self) -> str | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT MIN(next_fire_at) AS next FROM scheduled_events "
                "WHERE status = 'pending' AND next_fire_at IS NOT NULL"
            ).fetchone()
        return row["next"] if row else None

    def claim_event_for_fire(
        self,
        *,
        event_id: str,
        expected_next_fire_at: str,
        lease_token: str,
        lease_pid: int,
    ) -> JsonObject | None:
        """Atomic claim: returns the row iff it was still pending at expected_next_fire_at.

        The next_fire_at equality is an optimistic-lock version that makes double-fire
        impossible across the catch-up scan, the due loop, and concurrent processes."""
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                "UPDATE scheduled_events SET status = 'firing', lease_token = ?, "
                "lease_pid = ?, updated_at = ? "
                "WHERE id = ? AND status = 'pending' AND next_fire_at = ?",
                (lease_token, lease_pid, _now_iso(), event_id, expected_next_fire_at),
            )
            if cur.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM scheduled_events WHERE id = ?", (event_id,)
            ).fetchone()
            return dict(row) if row else None

    def complete_one_shot(self, event_id: str, *, fired_at: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE scheduled_events SET status = 'done', next_fire_at = NULL, "
                "last_fired_at = ?, fire_count = fire_count + 1, last_error = NULL, "
                "lease_token = NULL, lease_pid = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'firing'",
                (fired_at, _now_iso(), event_id),
            )

    def reschedule_recurring(
        self, event_id: str, *, next_fire_at: str, fired_at: str, last_error: str | None = None
    ) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE scheduled_events SET status = 'pending', next_fire_at = ?, "
                "last_fired_at = ?, fire_count = fire_count + 1, last_error = ?, "
                "lease_token = NULL, lease_pid = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'firing'",
                (next_fire_at, fired_at, last_error, _now_iso(), event_id),
            )

    def mark_event_failed(self, event_id: str, *, error: str, fired_at: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE scheduled_events SET status = 'failed', next_fire_at = NULL, "
                "last_fired_at = ?, fire_count = fire_count + 1, last_error = ?, "
                "lease_token = NULL, lease_pid = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'firing'",
                (fired_at, error, _now_iso(), event_id),
            )

    def recover_orphaned_firing(self) -> list[JsonObject]:
        """Events left 'firing' by a crash: surface as needs_review; never auto-replay."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_events WHERE status = 'firing'"
            ).fetchall()
            orphans = [dict(row) for row in rows]
            if orphans:
                conn.execute(
                    "UPDATE scheduled_events SET status = 'needs_review', next_fire_at = NULL, "
                    "last_error = 'interrupted mid-fire', lease_token = NULL, lease_pid = NULL, "
                    "updated_at = ? WHERE status = 'firing'",
                    (_now_iso(),),
                )
        return orphans

    def _find_by_id_or_query(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        item_id: str | None,
        query: str | None,
        query_column: str,
        open_status: bool,
    ) -> JsonObject:
        if item_id:
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise ValueError(f"{table} item not found: {item_id}")
            return _decode_possible_json(dict(row))

        needle = (_clean_optional(query) or "").casefold()
        sql = f"SELECT * FROM {table}"
        if open_status:
            sql += " WHERE status != 'done'"
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql).fetchall()
        if not needle and len(rows) == 1:
            return _decode_possible_json(dict(rows[0]))
        matches = [
            _decode_possible_json(dict(row))
            for row in rows
            if needle and needle in str(row[query_column]).casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"{table} item not found")
        raise ValueError(f"multiple matching {table} items; use id")

    def _list_due_rows(
        self,
        table: str,
        *,
        date_from: str | None,
        date_to: str | None,
        include_overdue: bool,
        include_completed: bool,
        important_only: bool,
        limit: int,
    ) -> list[JsonObject]:
        important_priorities = ("high", "urgent", "critical", "important")
        clauses: list[str] = []
        params: list[Any] = []
        due_clauses = ["due_date IS NOT NULL"]
        due_params: list[Any] = []
        if date_to:
            due_clauses.append("due_date <= ?")
            due_params.append(date_to)
        if date_from and not include_overdue:
            due_clauses.append("due_date >= ?")
            due_params.append(date_from)
        due_sql = " AND ".join(due_clauses)
        if important_only:
            placeholders = ", ".join("?" for _ in important_priorities)
            clauses.append(f"(({due_sql}) OR lower(priority) IN ({placeholders}))")
            params.extend(due_params)
            params.extend(important_priorities)
        else:
            clauses.append(due_sql)
            params.extend(due_params)
        if not include_completed:
            clauses.append("status != 'done'")
        sql = f"SELECT * FROM {table} WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY due_date, due_time IS NULL, due_time, created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_decode_possible_json(dict(row)) for row in rows]


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _add_days(date_text: str, days: int) -> str:
    try:
        date_value = datetime.fromisoformat(date_text).date()
    except ValueError:
        return date_text
    return (date_value + timedelta(days=max(0, days))).isoformat()


def _required_text(value: Any, label: str) -> str:
    text = _clean_optional(value)
    if text is None:
        raise ValueError(f"{label} is required")
    return text


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_tags(tags: Any) -> list[str]:
    if not tags:
        return []
    if not isinstance(tags, list):
        tags = [tags]
    return [str(tag).strip().lower() for tag in tags if str(tag).strip()]


def _preference_key(value: Any) -> str:
    text = _required_text(value, "preference key").lower()
    chars: list[str] = []
    last_was_separator = False
    for char in text:
        if ("a" <= char <= "z") or ("0" <= char <= "9"):
            chars.append(char)
            last_was_separator = False
        elif not last_was_separator:
            chars.append("_")
            last_was_separator = True
    key = "".join(chars).strip("_")
    if not key:
        raise ValueError("preference key must contain at least one letter or digit")
    return key[:64]


def _compact_line(value: Any, *, label: str, max_chars: int) -> str:
    text = " ".join(_required_text(value, label).split())
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 3)].rstrip() + "..."


def _compact_optional(value: Any, *, max_chars: int) -> str | None:
    text = _clean_optional(value)
    if text is None:
        return None
    return _compact_line(text, label="text", max_chars=max_chars)


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _compact_title(content: str) -> str:
    words = content.split()
    title = " ".join(words[:8])
    if len(words) > 8:
        title += "..."
    return title or "Note"


def _checklist_item(text: str) -> JsonObject:
    return {
        "id": _new_id("item"),
        "text": text,
        "done": False,
        "created_at": _now_iso(),
        "completed_at": None,
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    return json.loads(str(value))


def _row_values(row: JsonObject, *keys: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, (list, dict)):
            value = _json_dumps(value)
        values.append(value)
    return tuple(values)


def _decode_json_fields(row: JsonObject, *fields: str) -> JsonObject:
    for field in fields:
        row[field] = _json_loads(row.get(field)) or []
    return row


def _decode_possible_json(row: JsonObject) -> JsonObject:
    for field in ("tags", "items"):
        if field in row:
            row = _decode_json_fields(row, field)
    if "done" in row:
        row = _bool_fields(row, "done")
    return row


def _bool_fields(row: JsonObject, *fields: str) -> JsonObject:
    for field in fields:
        row[field] = bool(row.get(field))
    return row


def _filter_text_rows(
    rows: list[sqlite3.Row],
    columns: tuple[str, ...],
    needle: str,
    limit: int,
) -> list[JsonObject]:
    filtered: list[JsonObject] = []
    for row in rows:
        item = _decode_possible_json(dict(row))
        text = " ".join(str(item.get(column) or "") for column in columns).casefold()
        if needle and needle not in text:
            continue
        filtered.append(item)
        if len(filtered) >= max(1, limit):
            break
    return filtered


def _count(conn: sqlite3.Connection, table: str, where: str | None = None) -> int:
    sql = f"SELECT COUNT(*) AS count FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return int(conn.execute(sql).fetchone()["count"])
