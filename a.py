import asyncio
import time


async def fetch_data():
    await asyncio.sleep(2)
    print('what is this?')


async def main():
    task = asyncio.create_task(fetch_data())
    await asyncio.sleep(3)
    print(-1)
    task.cancel()
    print(0)
    try:
        print(1)
        await task
        print(2)
    except asyncio.CancelledError:
        print("fetch_data was canceled!")
asyncio.run(main())