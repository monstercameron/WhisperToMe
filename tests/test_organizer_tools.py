from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whispertome.organizer.store import OrganizerStore
from whispertome.organizer.tools import build_organization_tool_registry


class OrganizerStoreTests(unittest.TestCase):
    def test_core_organizer_items_persist_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = OrganizerStore(root)

            note = store.add_note(content="Call Sam about the prototype.", tags=["People"])
            reminder = store.add_reminder(title="Call Sam", due_date="2026-06-07")
            checklist = store.create_checklist(
                title="Demo prep",
                items=["Charge laptop", "Pack adapter"],
            )
            completed = store.complete_checklist_item(
                checklist_id=checklist["id"],
                item_text="charge",
            )
            event = store.add_itinerary_item(
                title="Prototype demo",
                date="2026-06-07",
                start_time="10:00",
                location="Lab",
            )
            project = store.create_project(name="WhisperToMe")
            task = store.add_task(
                title="Write organizer tests",
                due_date="2026-06-07",
                project_id=project["id"],
            )
            important_task = store.add_task(
                title="Escalate model latency",
                priority="high",
            )
            plan = store.save_daily_plan(
                date="2026-06-07",
                focus="Prototype readiness",
                items=["Run demo", "Review notes"],
            )
            decision = store.add_decision(
                title="Use SQLite",
                decision="Persist organizer data in SQLite.",
                project_id=project["id"],
            )
            person = store.add_person(name="Sam", relationship="tester")

            self.assertTrue((root / "organizer.sqlite").exists())

            reloaded = OrganizerStore(root)
            self.assertEqual(reloaded.list_notes(query="Sam")[0]["id"], note["id"])
            self.assertEqual(
                reloaded.list_reminders(due_before="2026-06-07")[0]["id"],
                reminder["id"],
            )
            self.assertTrue(completed["item"]["done"])
            self.assertEqual(
                reloaded.list_checklists(include_completed=False)[0]["items"][0]["text"],
                "Pack adapter",
            )
            self.assertEqual(reloaded.list_itinerary(date_from="2026-06-07")[0]["id"], event["id"])
            self.assertEqual(reloaded.list_tasks(query="organizer")[0]["id"], task["id"])
            self.assertEqual(reloaded.list_projects(query="Whisper")[0]["id"], project["id"])
            timely = reloaded.list_time_sensitive(
                date_from="2026-06-07",
                date_to="2026-06-07",
            )
            self.assertEqual(timely["counts"]["reminders"], 1)
            self.assertEqual(timely["counts"]["tasks"], 1)
            self.assertEqual(timely["counts"]["itinerary"], 1)
            important = reloaded.list_time_sensitive(
                date_from="2026-06-07",
                days_ahead=0,
                important_only=True,
            )
            self.assertIn(
                important_task["id"],
                {item["id"] for item in important["tasks"]},
            )
            self.assertEqual(
                reloaded.get_daily_plan(date="2026-06-07")["saved_plan"]["id"],
                plan["id"],
            )
            self.assertEqual(reloaded.list_decisions(query="SQLite")[0]["id"], decision["id"])
            self.assertEqual(reloaded.list_people(query="tester")[0]["id"], person["id"])

    def test_migrates_legacy_json_store_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "store.json").write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "id": "note_legacy",
                                "created_at": "2026-06-06T00:00:00+00:00",
                                "title": "Legacy note",
                                "content": "Migrated from JSON.",
                                "tags": ["legacy"],
                            }
                        ],
                        "checklists": [
                            {
                                "id": "list_legacy",
                                "created_at": "2026-06-06T00:00:00+00:00",
                                "title": "Legacy list",
                                "tags": [],
                                "items": [
                                    {
                                        "id": "item_legacy",
                                        "text": "Keep this item",
                                        "done": False,
                                        "created_at": "2026-06-06T00:00:00+00:00",
                                        "completed_at": None,
                                    }
                                ],
                            }
                        ],
                        "itinerary": [
                            {
                                "id": "event_legacy",
                                "created_at": "2026-06-06T00:00:00+00:00",
                                "title": "Legacy event",
                                "date": "2026-06-08",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = OrganizerStore(root)
            OrganizerStore(root)

            self.assertEqual(store.list_notes(query="migrated")[0]["id"], "note_legacy")
            self.assertEqual(store.list_checklists(query="Legacy")[0]["id"], "list_legacy")
            self.assertEqual(store.list_itinerary(date_from="2026-06-08")[0]["id"], "event_legacy")

    def test_preferences_persist_upsert_and_render_compact_prompt_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = OrganizerStore(root)

            saved = store.save_preference(
                key="Preferred Name",
                category="identity",
                preference="Call the user Marcus.",
                value="Marcus",
                evidence="computer call me marcus",
                priority=90,
            )
            updated = store.save_preference(
                key="preferred_name",
                category="identity",
                preference="Call the user Marc.",
                value="Marc",
                evidence="Actually, call me Marc.",
                priority=95,
            )

            self.assertEqual(saved["key"], "preferred_name")
            self.assertEqual(updated["id"], saved["id"])

            reloaded = OrganizerStore(root)
            preferences = reloaded.list_preferences()
            self.assertEqual(len(preferences), 1)
            self.assertEqual(preferences[0]["preference"], "Call the user Marc.")
            self.assertTrue(preferences[0]["active"])

            context = reloaded.preference_prompt_context(max_items=4, max_chars=200)
            self.assertIn("User preferences:", context)
            self.assertIn("- Call the user Marc.", context)
            self.assertNotIn("Actually, call me Marc.", context)


class OrganizationToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_all_organization_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_organization_tool_registry(Path(tmp))
            tool_names = {tool["name"] for tool in registry.tool_specs()}

            self.assertIn("notes_add", tool_names)
            self.assertIn("reminders_add", tool_names)
            self.assertIn("checklist_create", tool_names)
            self.assertIn("itinerary_add", tool_names)
            self.assertIn("tasks_add", tool_names)
            self.assertIn("projects_create", tool_names)
            self.assertIn("time_sensitive_check", tool_names)
            self.assertIn("daily_plan_get", tool_names)
            self.assertIn("decisions_add", tool_names)
            self.assertIn("people_add", tool_names)
            self.assertIn("preferences_save", tool_names)
            self.assertIn("preferences_list", tool_names)
            self.assertIn("system_volume_get", tool_names)
            self.assertIn("system_volume_set", tool_names)
            self.assertIn("system_volume_change", tool_names)
            self.assertIn("system_volume_mute", tool_names)
            self.assertIn("screen_brightness_get", tool_names)
            self.assertIn("screen_brightness_set", tool_names)
            self.assertIn("screen_brightness_change", tool_names)
            self.assertIn("agent_window_minimize", tool_names)
            self.assertIn("desktop_capture", tool_names)

            note_event = registry.execute(
                "notes_add",
                {"content": "Remember that the wake word is computer."},
            )
            preference_event = registry.execute(
                "preferences_save",
                {
                    "key": "units",
                    "category": "units",
                    "preference": "Use metric units by default.",
                    "value": "metric",
                },
            )
            preferences_event = registry.execute("preferences_list", {"query": "metric"})
            reminder_event = registry.execute(
                "reminders_add",
                {"title": "Test reminder", "due_date": "2026-06-07"},
            )
            list_event = registry.execute("notes_list", {"query": "wake word"})
            checklist_event = registry.execute(
                "checklist_create",
                {"title": "Manual test", "items": ["Start TUI", "Say wake word"]},
            )
            project_event = registry.execute("projects_create", {"name": "Organizer"})
            task_event = registry.execute(
                "tasks_add",
                {
                    "title": "Add SQLite tools",
                    "priority": "high",
                    "project_id": project_event.output["project"]["id"],
                },
            )
            daily_plan_event = registry.execute("daily_plan_get", {"date": "2026-06-07"})
            timely_event = registry.execute(
                "time_sensitive_check",
                {
                    "date_from": "2026-06-07",
                    "days_ahead": 0,
                    "important_only": True,
                },
            )
            decision_event = registry.execute(
                "decisions_add",
                {"title": "SQLite", "decision": "Use sqlite for organizer state."},
            )
            person_event = registry.execute("people_add", {"name": "Alex", "relationship": "PM"})
            summary_event = registry.execute("organization_summary", {})

            self.assertTrue(note_event.ok)
            self.assertTrue(preference_event.ok)
            self.assertEqual(
                preferences_event.output["preferences"][0]["preference"],
                "Use metric units by default.",
            )
            self.assertTrue(reminder_event.ok)
            self.assertEqual(list_event.output["notes"][0]["id"], note_event.output["note"]["id"])
            self.assertEqual(checklist_event.output["checklist"]["title"], "Manual test")
            self.assertEqual(task_event.output["task"]["title"], "Add SQLite tools")
            self.assertEqual(daily_plan_event.output["date"], "2026-06-07")
            self.assertTrue(timely_event.ok)
            self.assertEqual(timely_event.output["counts"]["reminders"], 1)
            self.assertEqual(timely_event.output["counts"]["tasks"], 1)
            self.assertEqual(decision_event.output["decision"]["title"], "SQLite")
            self.assertEqual(person_event.output["person"]["name"], "Alex")
            self.assertEqual(summary_event.output["projects"], 1)
            self.assertEqual(summary_event.output["active_preferences"], 1)


if __name__ == "__main__":
    unittest.main()
