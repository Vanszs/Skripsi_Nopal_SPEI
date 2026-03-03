# Perplexity MCP Configuration

## Setup
Perplexity MCP telah dikonfigurasi untuk workspace ini dengan konfigurasi berikut:

### Location
File: `.vscode/settings.json`

### Configuration Details
- **Server Name**: `perplexity`
- **Command**: `npx -y @modelcontextprotocol/server-perplexity`
- **API Key**: Configured via environment variable

### How to Use

1. **Reload VS Code Window**
   - Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
   - Type "Developer: Reload Window"
   - Press Enter

2. **Verify MCP is Loaded**
   - Open GitHub Copilot Chat
   - The Perplexity tools should now be available

3. **Available Perplexity Tools**
   - `perplexity_search`: Search for information using Perplexity AI
   - `perplexity_ask`: Ask questions to Perplexity AI

### Example Usage in Copilot Chat

```
@workspace Search using Perplexity: "latest research on Temporal Fusion Transformer for time series"
```

or

```
Can you use Perplexity to find recent papers about SPEI drought prediction with deep learning?
```

### Troubleshooting

**If Perplexity MCP doesn't load:**

1. Ensure Node.js is installed:
   ```powershell
   node --version
   npm --version
   ```

2. Check VS Code output panel:
   - View → Output
   - Select "GitHub Copilot Chat" from dropdown

3. Verify settings.json syntax is valid (no JSON errors)

**If you get API key errors:**
- Verify the API key is still valid at: https://www.perplexity.ai/settings/api
- Check for typos in the key

### API Key Security

⚠️ **Important**: The API key is stored in workspace settings. Consider:
- Adding `.vscode/settings.json` to `.gitignore` if not already there
- Using user settings instead of workspace settings for sensitive keys

### Alternative: User-Level Configuration

To configure at user level (recommended for security):
1. Open Command Palette: `Ctrl+Shift+P`
2. Type "Preferences: Open User Settings (JSON)"
3. Add the Perplexity MCP configuration there instead

---

**Configured on**: March 3, 2026
**Status**: ✅ Ready to use after VS Code reload
