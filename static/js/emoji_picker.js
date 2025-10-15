// Lightweight Emoji Picker for NeuroScan messaging
// - Renders a simple grid of emojis into a container (id: emojiPicker or emojiPickerChat)
// - Inserts selected emoji into the focused textarea (#messageInput)
// - Closes on outside click or Escape

(function(){
    const EMOJIS = [
        '😀','😁','😂','🤣','😊','😇','🙂','🙃','😉','😍',
        '😘','😗','😚','😋','😜','🤪','🤗','🤔','🤨','😐',
        '😶','😴','😭','😤','😡','🤒','🤕','🤢','🤮','🤧',
        '🫠','🫡','👍','👎','👏','🙏','💪','🤝','✨','🔥',
        '🧠','🫀','🫁','💊','🩺','🩸','🏥','📄','📎','🔒'
    ];

    function createPicker(container) {
        container.innerHTML = '';
        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(8, 1fr)';
        grid.style.gap = '6px';

        EMOJIS.forEach(emoji => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'emoji-btn';
            btn.textContent = emoji;
            btn.style.fontSize = '20px';
            btn.style.padding = '6px';
            btn.style.border = 'none';
            btn.style.background = 'transparent';
            btn.style.cursor = 'pointer';
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                insertEmojiAtCaret(emoji);
                // keep picker open for multiple inserts
            });
            grid.appendChild(btn);
        });

        container.appendChild(grid);
    }

    function findMessageInput() {
        // prefer visible textarea with id messageInput
        const el = document.getElementById('messageInput');
        if (el) return el;
        // fallback: first textarea in DOM
        return document.querySelector('textarea') || null;
    }

    function insertEmojiAtCaret(emoji) {
        const input = findMessageInput();
        if (!input) return;

        // For textarea/inputs
        const start = input.selectionStart || 0;
        const end = input.selectionEnd || 0;
        const value = input.value || '';

        const newValue = value.slice(0, start) + emoji + value.slice(end);
        input.value = newValue;

        // Move caret after inserted emoji
        const newPos = start + emoji.length;
        input.selectionStart = input.selectionEnd = newPos;

        // Trigger input events so other scripts update UI (send button enabling, autosize...)
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.focus();
    }

    function togglePickerById(pickerId) {
        const picker = document.getElementById(pickerId);
        if (!picker) return;
        if (picker.style.display === 'none' || getComputedStyle(picker).display === 'none') {
            picker.style.display = 'block';
            // render if empty
            if (!picker.dataset.rendered) {
                createPicker(picker);
                picker.dataset.rendered = '1';
            }
            // position near input if possible
            positionPicker(picker);
            // attach outside click listener
            setTimeout(() => {
                document.addEventListener('click', outsideClickHandler);
                document.addEventListener('keydown', escHandler);
            }, 0);
        } else {
            picker.style.display = 'none';
            document.removeEventListener('click', outsideClickHandler);
            document.removeEventListener('keydown', escHandler);
        }
    }

    function positionPicker(picker) {
        const input = findMessageInput();
        if (!input) return;
        // Compute bounding boxes
        const rect = input.getBoundingClientRect();
        const pickerRect = picker.getBoundingClientRect();

        // Place above input if enough space, otherwise below
        const spaceAbove = rect.top;
        const spaceBelow = window.innerHeight - rect.bottom;

        if (spaceAbove > pickerRect.height + 20) {
            picker.style.top = (window.scrollY + rect.top - pickerRect.height - 8) + 'px';
        } else {
            picker.style.top = (window.scrollY + rect.bottom + 8) + 'px';
        }

        // Align right edge with input right edge
        picker.style.left = Math.max(8, window.scrollX + rect.right - pickerRect.width) + 'px';
        picker.style.position = 'absolute';
    }

    function outsideClickHandler(e) {
        const pickers = document.querySelectorAll('#emojiPicker, #emojiPickerChat');
        for (const p of pickers) {
            if (p && p.style.display !== 'none' && !p.contains(e.target) && !e.target.closest('.btn-emoji') && !e.target.closest('#emojiBtn')) {
                p.style.display = 'none';
            }
        }
        document.removeEventListener('click', outsideClickHandler);
    }

    function escHandler(e) {
        if (e.key === 'Escape') {
            const pickers = document.querySelectorAll('#emojiPicker, #emojiPickerChat');
            pickers.forEach(p => { if (p) p.style.display = 'none'; });
            document.removeEventListener('keydown', escHandler);
        }
    }

    // Public toggles used by inline onclicks
    window.toggleEmojiPicker = function() {
        // prefer the page-specific picker id
        if (document.getElementById('emojiPicker')) {
            togglePickerById('emojiPicker');
        } else if (document.getElementById('emojiPickerChat')) {
            togglePickerById('emojiPickerChat');
        }
    };

    // Also support the chat template button id
    window.toggleEmojiPickerChat = function() {
        togglePickerById('emojiPickerChat');
    };

    // On DOM ready, ensure picker exists on chat template if present
    document.addEventListener('DOMContentLoaded', () => {
        const picker = document.getElementById('emojiPicker');
        const pickerChat = document.getElementById('emojiPickerChat');
        if (picker) {
            // initial render deferred to toggle
            picker.style.display = 'none';
        }
        if (pickerChat) pickerChat.style.display = 'none';
    });

})();
